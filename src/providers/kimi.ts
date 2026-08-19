/**
 * Kimi Code (Moonshot) subscription provider: RFC 8628 device-code OAuth
 * against auth.kimi.com, and streaming against the Kimi Coding Anthropic
 * Messages endpoint.
 */

import { EMPTY_RESPONSE_CODE, errorChain, LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { DeviceAuthorization } from '../auth/device-flow.js'
import type { KimiSession } from '../auth/store.js'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { resolveImages } from '../translate/resolved.js'
import {
  streamAnthropic,
  toAnthropicMessages,
  toAnthropicSystem,
  toAnthropicTools,
} from '../translate/anthropic.js'
import {
  httpLlmError,
  idleWatchdog,
  mapFetchFailure,
  ModelCatalogCache,
  oauthEndpointError,
  OAuthEndpointError,
  TokenManager,
} from './common.js'
import type { CatalogPersistence, DiscoveredModel, FetchFn, ModelEntry, ProviderUsage, UsageWindow } from './common.js'

export const KIMI_CLIENT_ID = '17e5f671-d194-4dfb-9706-5516cb48c098'
export const KIMI_OAUTH_HOST = 'https://auth.kimi.com'
export const KIMI_API_URL = 'https://api.kimi.com/coding/v1/messages'
export const KIMI_MODELS_URL = 'https://api.kimi.com/coding/v1/models'
export const KIMI_USAGE_URL = 'https://api.kimi.com/coding/v1/usages'
const KIMI_CONTEXT_WINDOW = 262_144
const KIMI_DEFAULT_MAX_TOKENS = 32_768
const KIMI_USER_AGENT = 'KimiCLI/1.5'
const DEFAULT_POLL_INTERVAL_SECONDS = 5
/** Refresh when the access token has less than this much life left. */
export const KIMI_PREEMPT_MS = 5 * 60_000

/** Token endpoint response shape (subset). */
interface KimiTokenResponse {
  access_token?: string
  refresh_token?: string
  expires_in?: number
  error?: string
  error_description?: string
  interval?: number
}

/** Build a session from a token response. */
function kimiSession(
  tokens: KimiTokenResponse,
  fallbackRefreshToken?: string,
  extras?: Pick<KimiSession, 'account' | 'plan'>,
): KimiSession {
  if (typeof tokens.access_token !== 'string' || tokens.access_token.length === 0) {
    throw new Error('kimi token endpoint returned no access token')
  }
  const refreshToken = tokens.refresh_token ?? fallbackRefreshToken
  if (refreshToken === undefined) throw new Error('kimi token endpoint returned no refresh token')
  if (typeof tokens.expires_in !== 'number' || tokens.expires_in <= 0) {
    throw new Error('kimi token endpoint returned no usable expiry')
  }
  return {
    accessToken: tokens.access_token,
    refreshToken,
    expiresAt: Date.now() + tokens.expires_in * 1000,
    ...extras,
  }
}

/** Only http(s) verification URLs are opened in the user's browser. */
function trustedHttpUrl(value: unknown): string | undefined {
  if (typeof value !== 'string' || value.length === 0) return undefined
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' && url.protocol !== 'http:') return undefined
    return url.href
  } catch {
    return undefined
  }
}

/**
 * Start a Kimi Code device-authorization grant.
 * @returns the device/user codes and verification URLs.
 */
export async function startKimiDeviceAuthorization(): Promise<DeviceAuthorization> {
  const response = await fetch(`${KIMI_OAUTH_HOST}/api/oauth/device_authorization`, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      accept: 'application/json',
    },
    body: new URLSearchParams({ client_id: KIMI_CLIENT_ID }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'kimi device authorization')
  const json = await response.json() as {
    device_code?: string
    user_code?: string
    verification_uri?: string
    verification_uri_complete?: string
    interval?: number
    expires_in?: number
  }
  const verificationUri = trustedHttpUrl(json.verification_uri)
  const verificationUriComplete = trustedHttpUrl(json.verification_uri_complete) ?? verificationUri
  if (typeof json.device_code !== 'string' || json.device_code.length === 0
    || typeof json.user_code !== 'string' || json.user_code.length === 0
    || verificationUri === undefined || verificationUriComplete === undefined) {
    throw new Error('kimi device authorization returned an incomplete response')
  }
  return {
    deviceCode: json.device_code,
    userCode: json.user_code,
    verificationUri,
    verificationUriComplete,
    intervalSeconds: typeof json.interval === 'number' && json.interval > 0
      ? json.interval
      : DEFAULT_POLL_INTERVAL_SECONDS,
    expiresInSeconds: typeof json.expires_in === 'number' && json.expires_in > 0
      ? json.expires_in
      : 15 * 60,
  }
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason instanceof Error ? signal.reason : new Error('login cancelled'))
      return
    }
    const timer = setTimeout(resolve, ms)
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(signal.reason instanceof Error ? signal.reason : new Error('login cancelled'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * Poll the Kimi token endpoint until the user approves the device grant.
 * @param device - the authorization started by {@link startKimiDeviceAuthorization}.
 * @param signal - abort when the attempt is cancelled or times out.
 * @returns the session to store.
 */
export async function pollKimiDeviceToken(
  device: DeviceAuthorization,
  signal: AbortSignal,
): Promise<KimiSession> {
  let intervalMs = device.intervalSeconds * 1000
  // RFC 8628: wait one interval before the first poll.
  await sleep(intervalMs, signal)
  while (!signal.aborted) {
    const response = await fetch(`${KIMI_OAUTH_HOST}/api/oauth/token`, {
      method: 'POST',
      headers: {
        'content-type': 'application/x-www-form-urlencoded',
        accept: 'application/json',
      },
      body: new URLSearchParams({
        client_id: KIMI_CLIENT_ID,
        device_code: device.deviceCode,
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
      }).toString(),
      signal,
    })
    const json = await response.json().catch(() => ({})) as KimiTokenResponse
    if (response.ok && typeof json.access_token === 'string') {
      return kimiSession(json)
    }
    const error = json.error
    if (error === 'authorization_pending') {
      await sleep(intervalMs, signal)
      continue
    }
    if (error === 'slow_down') {
      if (typeof json.interval === 'number' && json.interval > 0) intervalMs = json.interval * 1000
      else intervalMs += 5000
      await sleep(intervalMs, signal)
      continue
    }
    if (error === 'expired_token') throw new Error('Kimi Code device authorization expired. Please restart login.')
    if (error === 'access_denied') throw new Error('Kimi Code login was denied.')
    if (response.status >= 500) throw await oauthEndpointError(response, 'kimi')
    const detail = json.error_description ?? error ?? `HTTP ${String(response.status)}`
    throw new Error(`Kimi Code device token request failed: ${detail}`)
  }
  throw signal.reason instanceof Error ? signal.reason : new Error('login cancelled')
}

/**
 * Refresh a kimi session (form-encoded grant). The refresh token rotates.
 * @param session - the stored session.
 * @returns the fresh session to store.
 */
export async function refreshKimi(session: KimiSession): Promise<KimiSession> {
  const response = await fetch(`${KIMI_OAUTH_HOST}/api/oauth/token`, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      accept: 'application/json',
    },
    body: new URLSearchParams({
      client_id: KIMI_CLIENT_ID,
      grant_type: 'refresh_token',
      refresh_token: session.refreshToken,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'kimi')
  const next = kimiSession(await response.json() as KimiTokenResponse, session.refreshToken)
  return {
    ...next,
    ...session.account === undefined ? {} : { account: session.account },
    ...session.plan === undefined ? {} : { plan: session.plan },
  }
}

/**
 * Whether a kimi refresh failure means the login is permanently gone.
 * @param error - the thrown refresh error.
 * @returns true when re-login is the only fix.
 */
export function isKimiPermanentRefreshError(error: unknown): boolean {
  return error instanceof OAuthEndpointError
    && (error.status === 401 || error.status === 403 || error.oauthCode === 'invalid_grant')
}

/** RFC3339 `resetTime` value → epoch ms, or undefined when absent/unparsable. */
function kimiResetsAt(value: unknown): number | undefined {
  if (typeof value !== 'string' || value.length === 0) return undefined
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

/** Map one `{limit, remaining}` bucket into a used-percent window. */
function kimiWindow(
  detail: { limit?: string; remaining?: string; resetTime?: string } | undefined,
  kind: UsageWindow['kind'],
): UsageWindow | undefined {
  if (detail === undefined) return undefined
  const limit = Number(detail.limit)
  const remaining = Number(detail.remaining)
  if (!Number.isFinite(limit) || limit <= 0 || !Number.isFinite(remaining)) return undefined
  const usedPercent = Math.max(0, Math.min(100, ((limit - remaining) / limit) * 100))
  const resetsAt = kimiResetsAt(detail.resetTime)
  return { kind, usedPercent, ...resetsAt === undefined ? {} : { resetsAt } }
}

const KIMI_PLAN_NAMES: Record<string, string> = {
  LEVEL_FREE: 'Free',
  LEVEL_BASIC: 'Basic',
  LEVEL_INTERMEDIATE: 'Pro',
  LEVEL_ADVANCED: 'Max',
}

/**
 * Fetch Kimi Code subscription usage from `/v1/usages`.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @param signal - caller cancellation.
 * @returns mapped usage windows and plan name.
 */
export async function fetchKimiUsage(
  session: KimiSession,
  fetchFn: FetchFn = fetch,
  signal?: AbortSignal,
): Promise<ProviderUsage> {
  const response = await fetchFn(KIMI_USAGE_URL, {
    headers: {
      authorization: `Bearer ${session.accessToken}`,
      accept: 'application/json',
      'user-agent': KIMI_USER_AGENT,
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) throw await httpLlmError(response, 'kimi usage API')
  const payload = await response.json() as {
    usage?: { limit?: string; remaining?: string; resetTime?: string }
    limits?: Array<{
      window?: { duration?: number; timeUnit?: string }
      detail?: { limit?: string; remaining?: string; resetTime?: string }
    }>
    user?: { membership?: { level?: string } }
  }
  const windows: UsageWindow[] = []
  const overall = kimiWindow(payload.usage, 'weekly')
  if (overall !== undefined) windows.push(overall)
  for (const entry of payload.limits ?? []) {
    const kind = entry.window?.duration === 300 ? 'session' : 'other'
    const window = kimiWindow(entry.detail, kind)
    if (window !== undefined) windows.push(window)
  }
  const level = payload.user?.membership?.level
  const plan = typeof level === 'string' && level.length > 0
    ? (KIMI_PLAN_NAMES[level] ?? level)
    : undefined
  return {
    supported: true,
    windows,
    ...plan === undefined ? {} : { plan },
  }
}

/** The Kimi 2.x / 3 family accepts image input. */
const KIMI_MODALITIES: readonly ('text' | 'image')[] = ['text', 'image']

const KIMI_REASONING: NonNullable<DiscoveredModel['reasoning']> = {
  efforts: [
    { id: ReasoningEffortId('low'), name: 'Low' },
    { id: ReasoningEffortId('high'), name: 'High' },
    { id: ReasoningEffortId('max'), name: 'Max' },
  ],
}

/** Built-in catalog used when live discovery is unavailable. */
export const KIMI_DEFAULT_MODELS: ModelEntry[] = [
  { id: 'k3', name: 'Kimi K3', contextWindow: 1_048_576, maxTokens: 131_072, inputModalities: [...KIMI_MODALITIES] },
  { id: 'k3-256k', name: 'Kimi K3-256K', contextWindow: 262_144, maxTokens: 131_072, inputModalities: [...KIMI_MODALITIES] },
  { id: 'kimi-for-coding', name: 'Kimi K2.7 Code', contextWindow: 262_144, maxTokens: 32_768, inputModalities: [...KIMI_MODALITIES] },
  { id: 'kimi-for-coding-highspeed', name: 'Kimi For Coding HighSpeed', contextWindow: 262_144, maxTokens: 32_768, inputModalities: [...KIMI_MODALITIES] },
]

/**
 * Fetch the live Kimi Coding model catalog.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @returns discovered models in endpoint order.
 */
export async function fetchKimiModels(
  session: KimiSession,
  fetchFn: FetchFn = fetch,
): Promise<DiscoveredModel[]> {
  const response = await fetchFn(KIMI_MODELS_URL, {
    headers: {
      authorization: `Bearer ${session.accessToken}`,
      accept: 'application/json',
      'user-agent': KIMI_USER_AGENT,
    },
  })
  if (!response.ok) throw await httpLlmError(response, 'kimi models API')
  const payload = await response.json() as {
    data?: Array<{ id?: string; display_name?: string; name?: string }>
  }
  const source = Array.isArray(payload.data) ? payload.data : []
  const seen = new Set<string>()
  const models: DiscoveredModel[] = []
  for (const entry of source) {
    if (typeof entry.id !== 'string' || entry.id.length === 0 || seen.has(entry.id)) continue
    seen.add(entry.id)
    const name = (typeof entry.display_name === 'string' && entry.display_name.length > 0)
      ? entry.display_name
      : (typeof entry.name === 'string' && entry.name.length > 0 ? entry.name : entry.id)
    models.push({ id: entry.id, name, reasoning: KIMI_REASONING, thinkingType: 'adaptive' })
  }
  if (models.length === 0) throw new Error('kimi models API returned an empty catalog')
  return models
}

/** Constructor dependencies for {@link KimiAdapter}. */
export interface KimiAdapterOptions {
  models: readonly ModelEntry[]
  streamIdleTimeoutMs: number
  tokens: TokenManager<KimiSession>
  /** Whether to fetch the live catalog when logged in (false when config `models` overrides). */
  discovery: boolean
  fetchFn?: FetchFn
  onWarn?: (message: string) => void
  /** Resolve the attachment service per request; absent means image requests fail loudly. */
  resolveAttachments?: () => AttachmentStore | undefined
  /** Durable catalog store seeding capability metadata across restarts. */
  catalogStore?: CatalogPersistence
}

/** Kimi Code wire adapter: one instance serves the `kimi` provider route. */
export class KimiAdapter extends LlmAdapter {
  private readonly catalog: ModelCatalogCache

  constructor(private readonly options: KimiAdapterOptions) {
    super()
    this.catalog = new ModelCatalogCache(options.catalogStore)
  }

  private async fetchCatalog(): Promise<DiscoveredModel[]> {
    return fetchKimiModels(await this.options.tokens.session(), this.options.fetchFn)
  }

  private async discovered(model: string): Promise<DiscoveredModel | undefined> {
    if (!this.options.discovery) return undefined
    const models = await this.catalog.resolve(() => this.fetchCatalog())
    return models?.find(entry => entry.id === model)
  }

  private staticModels(provider: string): LlmModelInfo[] {
    return this.options.models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: model.inputModalities ?? KIMI_MODALITIES,
    }))
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Kimi Code (Subscription)' }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    if (await this.options.tokens.peek() === undefined) return []
    if (!this.options.discovery) return this.staticModels(provider)
    try {
      const models = await this.catalog.get(() => this.fetchCatalog())
      return models.map(model => ({
        provider,
        id: model.id,
        name: model.name,
        inputModalities: KIMI_MODALITIES,
      }))
    } catch (error: unknown) {
      if (error instanceof LlmError
        && (error.code === 'MISSING_CREDENTIAL' || error.code === 'INVALID_CREDENTIAL')) return []
      if (error instanceof LlmError && error.code === 'AUTH') this.catalog.invalidate()
      this.options.onWarn?.(
        `kimi model discovery failed; using the built-in catalog (${errorChain(error)})`,
      )
      return this.staticModels(provider)
    }
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const disc = await this.discovered(model)
    const configured = this.options.models.find(entry => entry.id === model)
    return {
      provider,
      id: model,
      name: disc?.name ?? configured?.name ?? model,
      inputModalities: configured?.inputModalities ?? KIMI_MODALITIES,
      context: {
        contextWindow: disc?.contextWindow ?? configured?.contextWindow ?? KIMI_CONTEXT_WINDOW,
      },
      defaultMaxTokens: configured?.maxTokens ?? KIMI_DEFAULT_MAX_TOKENS,
      reasoning: disc?.reasoning ?? KIMI_REASONING,
    }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const watchdog = idleWatchdog(options.signal, this.options.streamIdleTimeoutMs)
    try {
      let session = await this.options.tokens.session()
      let response = await this.request(options, session, watchdog.signal)
      if (response.status === 401) {
        session = await this.options.tokens.session(true)
        response = await this.request(options, session, watchdog.signal)
      }
      if (!response.ok) throw await httpLlmError(response, 'kimi API')
      if (response.body === null) {
        throw new LlmError('kimi API returned no response body', EMPTY_RESPONSE_CODE)
      }
      yield* streamAnthropic(response.body, () => { watchdog.pulse() })
    } catch (error: unknown) {
      throw mapFetchFailure('kimi API', error, watchdog, options.signal)
    } finally {
      watchdog.stop()
    }
  }

  private thinkingParam(maxTokens: number): Record<string, unknown> {
    if (maxTokens >= 2_048) return { type: 'adaptive', display: 'summarized' }
    return { type: 'enabled', budget_tokens: Math.max(1_024, Math.floor(maxTokens * 0.5)), display: 'summarized' }
  }

  private async request(options: GenerateOptions, session: KimiSession, signal: AbortSignal): Promise<Response> {
    const messages = await resolveImages(options.messages, this.options.resolveAttachments?.(), signal)
    const maxTokens = options.maxTokens
      ?? this.options.models.find(entry => entry.id === options.model)?.maxTokens
      ?? KIMI_DEFAULT_MAX_TOKENS
    const body = {
      model: options.model,
      max_tokens: maxTokens,
      system: toAnthropicSystem(options.system, messages, ''),
      messages: toAnthropicMessages(messages),
      ...options.tools !== undefined && options.tools.length > 0
        ? { tools: toAnthropicTools(options.tools) }
        : {},
      thinking: this.thinkingParam(maxTokens),
      ...options.reasoningEffort !== undefined
        ? { output_config: { effort: String(options.reasoningEffort) } }
        : {},
      stream: true,
    }
    return fetch(KIMI_API_URL, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${session.accessToken}`,
        'anthropic-version': '2023-06-01',
        'user-agent': KIMI_USER_AGENT,
        accept: 'text/event-stream',
        'content-type': 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    })
  }
}
