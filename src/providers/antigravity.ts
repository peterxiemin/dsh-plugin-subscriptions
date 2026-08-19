/**
 * Antigravity (Google) subscription provider: OAuth against accounts.google.com
 * with the Antigravity IDE client, and streaming against the Cloud Code Assist
 * Gemini-style gateway (`cloudcode-pa.googleapis.com`).
 */

import { randomUUID } from 'node:crypto'
import { EMPTY_RESPONSE_CODE, errorChain, LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { FlowSpec } from '../auth/oauth-flow.js'
import type { AntigravitySession } from '../auth/store.js'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { resolveImages } from '../translate/resolved.js'
import { streamGemini, toGeminiContents, toGeminiSystem, toGeminiTools } from '../translate/gemini.js'
import {
  httpLlmError,
  idleWatchdog,
  mapFetchFailure,
  ModelCatalogCache,
  oauthEndpointError,
  OAuthEndpointError,
  TokenManager,
} from './common.js'
import type { CatalogPersistence, DiscoveredModel, FetchFn, ModelEntry, ProviderUsage } from './common.js'

// Public Antigravity IDE OAuth client (desktop app credentials, not a user secret).
export const ANTIGRAVITY_CLIENT_ID = [
  '1071006060591-tmhssin2h21lcre235vtolojh4g403ep',
  'apps.googleusercontent.com',
].join('.')
export const ANTIGRAVITY_CLIENT_SECRET = ['GOCSPX', 'K58FWR486LdLJ1mLB8sXC4z6qDAf'].join('-')
export const ANTIGRAVITY_AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
export const ANTIGRAVITY_TOKEN_URL = 'https://oauth2.googleapis.com/token'
export const ANTIGRAVITY_USERINFO_URL = 'https://www.googleapis.com/oauth2/v1/userinfo?alt=json'
export const ANTIGRAVITY_ENDPOINT = 'https://cloudcode-pa.googleapis.com'
export const ANTIGRAVITY_DEFAULT_PROJECT_ID = 'rising-fact-p41fc'
const ANTIGRAVITY_SCOPES = [
  'https://www.googleapis.com/auth/cloud-platform',
  'https://www.googleapis.com/auth/userinfo.email',
  'https://www.googleapis.com/auth/userinfo.profile',
  'https://www.googleapis.com/auth/cclog',
  'https://www.googleapis.com/auth/experimentsandconfigs',
].join(' ')
const ANTIGRAVITY_CALLBACK_PATH = '/oauth-callback'
const ANTIGRAVITY_CONTEXT_WINDOW = 1_048_576
const ANTIGRAVITY_DEFAULT_MAX_TOKENS = 65_536
const ANTIGRAVITY_VERSION = '1.15.8'
/** Refresh when the access token has less than this much life left. */
export const ANTIGRAVITY_PREEMPT_MS = 5 * 60_000

function antigravityHeaders(): Record<string, string> {
  const platform = process.platform === 'win32' ? 'WINDOWS' : 'MACOS'
  return {
    'user-agent': `antigravity/${ANTIGRAVITY_VERSION} ${process.platform === 'darwin' ? 'darwin/arm64' : 'windows/amd64'}`,
    'x-goog-api-client': 'google-cloud-sdk vscode_cloudshelleditor/0.1',
    'client-metadata': JSON.stringify({
      ideType: 'ANTIGRAVITY',
      platform,
      pluginType: 'GEMINI',
    }),
  }
}

/** Static antigravity flow facts for the OAuth flow engine. */
export const antigravityFlow: FlowSpec = {
  callbackPath: ANTIGRAVITY_CALLBACK_PATH,
  listen: { host: 'localhost', ports: [51121] },
  timeoutMs: 300_000,
  buildAuthorizeUrl({ redirectUri, state, pkce }) {
    const params = new URLSearchParams({
      client_id: ANTIGRAVITY_CLIENT_ID,
      response_type: 'code',
      redirect_uri: redirectUri,
      scope: ANTIGRAVITY_SCOPES,
      code_challenge: pkce.challenge,
      code_challenge_method: 'S256',
      state,
      access_type: 'offline',
      prompt: 'consent',
    })
    return `${ANTIGRAVITY_AUTHORIZE_URL}?${params.toString()}`
  },
}

/** Token endpoint response shape (subset). */
interface AntigravityTokenResponse {
  access_token?: string
  refresh_token?: string
  expires_in?: number
}

/** Best-effort Google userinfo; login must not fail when this does. */
async function fetchAntigravityEmail(accessToken: string): Promise<string | undefined> {
  try {
    const response = await fetch(ANTIGRAVITY_USERINFO_URL, {
      headers: { authorization: `Bearer ${accessToken}` },
    })
    if (!response.ok) return undefined
    const profile = await response.json() as { email?: string }
    return typeof profile.email === 'string' && profile.email.length > 0 ? profile.email : undefined
  } catch {
    return undefined
  }
}

/** Resolve the Cloud Code Assist project id via `loadCodeAssist`. */
async function fetchAntigravityProjectId(accessToken: string): Promise<string> {
  try {
    const response = await fetch(`${ANTIGRAVITY_ENDPOINT}/v1internal:loadCodeAssist`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${accessToken}`,
        'content-type': 'application/json',
        ...antigravityHeaders(),
      },
      body: JSON.stringify({
        metadata: {
          ideType: 'ANTIGRAVITY',
          platform: process.platform === 'win32' ? 'WINDOWS' : 'MACOS',
          pluginType: 'GEMINI',
        },
      }),
    })
    if (!response.ok) return ANTIGRAVITY_DEFAULT_PROJECT_ID
    const data = await response.json() as {
      cloudaicompanionProject?: string | { id?: string }
    }
    if (typeof data.cloudaicompanionProject === 'string' && data.cloudaicompanionProject.length > 0) {
      return data.cloudaicompanionProject
    }
    if (typeof data.cloudaicompanionProject === 'object' && data.cloudaicompanionProject !== null
      && typeof data.cloudaicompanionProject.id === 'string'
      && data.cloudaicompanionProject.id.length > 0) {
      return data.cloudaicompanionProject.id
    }
  } catch {
    // Project lookup is best-effort; the default project still works for consumer accounts.
  }
  return ANTIGRAVITY_DEFAULT_PROJECT_ID
}

/** Build a session from a token response. */
async function antigravitySession(
  tokens: AntigravityTokenResponse,
  fallback: { refreshToken?: string; projectId?: string; emailAddress?: string },
  withProfile: boolean,
): Promise<AntigravitySession> {
  if (typeof tokens.access_token !== 'string' || tokens.access_token.length === 0) {
    throw new Error('antigravity token endpoint returned no access token')
  }
  const refreshToken = tokens.refresh_token ?? fallback.refreshToken
  if (refreshToken === undefined) throw new Error('antigravity token endpoint returned no refresh token')
  if (typeof tokens.expires_in !== 'number' || tokens.expires_in <= 0) {
    throw new Error('antigravity token endpoint returned no usable expiry')
  }
  const emailAddress = withProfile
    ? await fetchAntigravityEmail(tokens.access_token)
    : fallback.emailAddress
  const projectId = fallback.projectId !== undefined && fallback.projectId.length > 0
    ? fallback.projectId
    : await fetchAntigravityProjectId(tokens.access_token)
  return {
    accessToken: tokens.access_token,
    refreshToken,
    expiresAt: Date.now() + tokens.expires_in * 1000,
    projectId,
    ...emailAddress === undefined ? {} : { emailAddress },
  }
}

/**
 * Exchange an authorization code for an antigravity session (form-encoded grant).
 * @param code - the authorization code from the callback.
 * @param verifier - the PKCE verifier minted for the attempt.
 * @param redirectUri - the attempt's redirect URI.
 * @returns the session to store.
 */
export async function exchangeAntigravityCode(
  code: string,
  verifier: string,
  redirectUri: string,
): Promise<AntigravitySession> {
  const response = await fetch(ANTIGRAVITY_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: ANTIGRAVITY_CLIENT_ID,
      client_secret: ANTIGRAVITY_CLIENT_SECRET,
      code,
      grant_type: 'authorization_code',
      redirect_uri: redirectUri,
      code_verifier: verifier,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'antigravity')
  return antigravitySession(await response.json() as AntigravityTokenResponse, {}, true)
}

/**
 * Refresh an antigravity session (form-encoded grant).
 * @param session - the stored session.
 * @returns the fresh session to store.
 */
export async function refreshAntigravity(session: AntigravitySession): Promise<AntigravitySession> {
  const response = await fetch(ANTIGRAVITY_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: ANTIGRAVITY_CLIENT_ID,
      client_secret: ANTIGRAVITY_CLIENT_SECRET,
      grant_type: 'refresh_token',
      refresh_token: session.refreshToken,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'antigravity')
  return antigravitySession(
    await response.json() as AntigravityTokenResponse,
    {
      refreshToken: session.refreshToken,
      projectId: session.projectId,
      ...session.emailAddress === undefined ? {} : { emailAddress: session.emailAddress },
    },
    false,
  )
}

/**
 * Whether an antigravity refresh failure means the login is permanently gone.
 * @param error - the thrown refresh error.
 * @returns true when re-login is the only fix.
 */
export function isAntigravityPermanentRefreshError(error: unknown): boolean {
  return error instanceof OAuthEndpointError
    && (error.status === 400 || error.status === 401)
    && (error.oauthCode === 'invalid_grant' || error.oauthCode === 'invalid_token' || error.oauthCode === undefined)
}

/** Antigravity has no documented usage endpoint. */
export async function fetchAntigravityUsage(): Promise<ProviderUsage> {
  return { supported: false }
}

const ANTIGRAVITY_MODALITIES: readonly ('text' | 'image')[] = ['text', 'image']

const GEMINI_REASONING: NonNullable<DiscoveredModel['reasoning']> = {
  efforts: [
    { id: ReasoningEffortId('low'), name: 'Low' },
    { id: ReasoningEffortId('medium'), name: 'Medium' },
    { id: ReasoningEffortId('high'), name: 'High' },
  ],
  defaultEffort: ReasoningEffortId('high'),
}

/** Built-in catalog used when live discovery is unavailable. */
export const ANTIGRAVITY_DEFAULT_MODELS: ModelEntry[] = [
  { id: 'gemini-3-pro-high', name: 'Gemini 3 Pro High', contextWindow: 1_048_576, maxTokens: 65_535, inputModalities: [...ANTIGRAVITY_MODALITIES] },
  { id: 'gemini-3-pro-low', name: 'Gemini 3 Pro Low', contextWindow: 1_048_576, maxTokens: 65_535, inputModalities: [...ANTIGRAVITY_MODALITIES] },
  { id: 'gemini-3-flash', name: 'Gemini 3 Flash', contextWindow: 1_048_576, maxTokens: 65_536, inputModalities: [...ANTIGRAVITY_MODALITIES] },
  { id: 'claude-opus-4-6-thinking', name: 'Claude Opus 4.6 Thinking', contextWindow: 200_000, maxTokens: 64_000, inputModalities: [...ANTIGRAVITY_MODALITIES] },
  { id: 'claude-sonnet-4-5-thinking', name: 'Claude Sonnet 4.5 Thinking', contextWindow: 200_000, maxTokens: 64_000, inputModalities: [...ANTIGRAVITY_MODALITIES] },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5', contextWindow: 200_000, maxTokens: 64_000, inputModalities: [...ANTIGRAVITY_MODALITIES] },
]

/**
 * Fetch the live Antigravity model catalog from `loadCodeAssist`.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @returns discovered models, or throws when the payload is unusable.
 */
export async function fetchAntigravityModels(
  session: AntigravitySession,
  fetchFn: FetchFn = fetch,
): Promise<DiscoveredModel[]> {
  const response = await fetchFn(`${ANTIGRAVITY_ENDPOINT}/v1internal:loadCodeAssist`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${session.accessToken}`,
      'content-type': 'application/json',
      accept: 'application/json',
      ...antigravityHeaders(),
    },
    body: JSON.stringify({
      metadata: {
        ideType: 'ANTIGRAVITY',
        platform: process.platform === 'win32' ? 'WINDOWS' : 'MACOS',
        pluginType: 'GEMINI',
      },
    }),
  })
  if (!response.ok) throw await httpLlmError(response, 'antigravity models API')
  const payload = await response.json() as {
    allowedModels?: Array<{ name?: string; displayName?: string; modelId?: string }>
    models?: Array<{ name?: string; displayName?: string; id?: string }>
  }
  const source = payload.allowedModels ?? payload.models ?? []
  const seen = new Set<string>()
  const models: DiscoveredModel[] = []
  for (const entry of source) {
    const raw = entry as { modelId?: string; id?: string; name?: string; displayName?: string }
    const id = raw.modelId ?? raw.id ?? raw.name
    if (typeof id !== 'string' || id.length === 0 || seen.has(id)) continue
    seen.add(id)
    const name = (typeof raw.displayName === 'string' && raw.displayName.length > 0)
      ? raw.displayName
      : id
    models.push({
      id,
      name,
      .../gemini-3|thinking/i.test(id) ? { reasoning: GEMINI_REASONING } : {},
    })
  }
  if (models.length === 0) throw new Error('antigravity models API returned an empty catalog')
  return models
}

/** Constructor dependencies for {@link AntigravityAdapter}. */
export interface AntigravityAdapterOptions {
  models: readonly ModelEntry[]
  streamIdleTimeoutMs: number
  tokens: TokenManager<AntigravitySession>
  /** Whether to fetch the live catalog when logged in (false when config `models` overrides). */
  discovery: boolean
  fetchFn?: FetchFn
  onWarn?: (message: string) => void
  /** Resolve the attachment service per request; absent means image requests fail loudly. */
  resolveAttachments?: () => AttachmentStore | undefined
  /** Durable catalog store seeding capability metadata across restarts. */
  catalogStore?: CatalogPersistence
}

/** Antigravity wire adapter: one instance serves the `antigravity` provider route. */
export class AntigravityAdapter extends LlmAdapter {
  private readonly catalog: ModelCatalogCache

  constructor(private readonly options: AntigravityAdapterOptions) {
    super()
    this.catalog = new ModelCatalogCache(options.catalogStore)
  }

  private async fetchCatalog(): Promise<DiscoveredModel[]> {
    return fetchAntigravityModels(await this.options.tokens.session(), this.options.fetchFn)
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
      inputModalities: model.inputModalities ?? ANTIGRAVITY_MODALITIES,
    }))
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Antigravity (Google)' }
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
        inputModalities: ANTIGRAVITY_MODALITIES,
      }))
    } catch (error: unknown) {
      if (error instanceof LlmError
        && (error.code === 'MISSING_CREDENTIAL' || error.code === 'INVALID_CREDENTIAL')) return []
      if (error instanceof LlmError && error.code === 'AUTH') this.catalog.invalidate()
      this.options.onWarn?.(
        `antigravity model discovery failed; using the built-in catalog (${errorChain(error)})`,
      )
      return this.staticModels(provider)
    }
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const disc = await this.discovered(model)
    const configured = this.options.models.find(entry => entry.id === model)
    const reasoning = disc?.reasoning
      ?? (/gemini-3|thinking/i.test(model) ? GEMINI_REASONING : undefined)
    return {
      provider,
      id: model,
      name: disc?.name ?? configured?.name ?? model,
      inputModalities: configured?.inputModalities ?? ANTIGRAVITY_MODALITIES,
      context: {
        contextWindow: disc?.contextWindow ?? configured?.contextWindow ?? ANTIGRAVITY_CONTEXT_WINDOW,
      },
      defaultMaxTokens: configured?.maxTokens ?? ANTIGRAVITY_DEFAULT_MAX_TOKENS,
      ...reasoning === undefined ? {} : { reasoning },
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
      if (!response.ok) throw await httpLlmError(response, 'antigravity API')
      if (response.body === null) {
        throw new LlmError('antigravity API returned no response body', EMPTY_RESPONSE_CODE)
      }
      yield* streamGemini(response.body, () => { watchdog.pulse() })
    } catch (error: unknown) {
      throw mapFetchFailure('antigravity API', error, watchdog, options.signal)
    } finally {
      watchdog.stop()
    }
  }

  private thinkingConfig(model: string, effort: string | undefined): Record<string, unknown> | undefined {
    const thinking = /gemini-3|thinking/i.test(model)
    if (!thinking) return undefined
    if (/gemini-3/.test(model)) {
      return {
        includeThoughts: true,
        thinkingLevel: effort ?? 'high',
      }
    }
    const budgets: Record<string, number> = { low: 8192, medium: 16_384, high: 32_768 }
    return {
      includeThoughts: true,
      thinkingBudget: budgets[effort ?? 'high'] ?? 16_384,
    }
  }

  private async request(options: GenerateOptions, session: AntigravitySession, signal: AbortSignal): Promise<Response> {
    const messages = await resolveImages(options.messages, this.options.resolveAttachments?.(), signal)
    const maxTokens = options.maxTokens
      ?? this.options.models.find(entry => entry.id === options.model)?.maxTokens
      ?? ANTIGRAVITY_DEFAULT_MAX_TOKENS
    const systemInstruction = toGeminiSystem(options.system, messages)
    const thinkingConfig = this.thinkingConfig(options.model, options.reasoningEffort === undefined
      ? undefined
      : String(options.reasoningEffort))
    const body = {
      project: session.projectId,
      model: options.model,
      userAgent: 'antigravity',
      requestId: randomUUID(),
      request: {
        contents: toGeminiContents(messages),
        generationConfig: {
          maxOutputTokens: maxTokens,
          ...thinkingConfig === undefined ? {} : { thinkingConfig },
        },
        ...systemInstruction === undefined ? {} : { systemInstruction },
        ...options.tools !== undefined && options.tools.length > 0
          ? { tools: toGeminiTools(options.tools) }
          : {},
      },
    }
    return fetch(`${ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${session.accessToken}`,
        accept: 'text/event-stream',
        'content-type': 'application/json',
        ...antigravityHeaders(),
      },
      body: JSON.stringify(body),
      signal,
    })
  }
}
