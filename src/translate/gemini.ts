/**
 * Translate between the harness message vocabulary and the Gemini / Cloud
 * Code Assist wire format used by Antigravity: request assembly (contents,
 * systemInstruction, functionDeclarations) and a push-model SSE translator.
 */

import { CallId, EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, StreamChunk, TokenUsage, ToolResultBlock, ToolSchema } from '@deepseek-ai/dsh-llm'
import { parseSse } from './sse.js'
import type { TranslatableMessage } from './resolved.js'

/** One Gemini `contents[]` entry. */
export interface GeminiContent {
  role: 'user' | 'model'
  parts: Record<string, unknown>[]
}

/** JSON Schema fields the Antigravity protobuf validator rejects. */
const UNSUPPORTED_SCHEMA_FIELDS = new Set([
  'additionalProperties',
  '$schema',
  '$id',
  '$comment',
  '$ref',
  '$defs',
  'definitions',
  'const',
  'contentMediaType',
  'contentEncoding',
  'if',
  'then',
  'else',
  'not',
  'patternProperties',
  'unevaluatedProperties',
  'unevaluatedItems',
  'dependentRequired',
  'dependentSchemas',
  'propertyNames',
  'minContains',
  'maxContains',
  'default',
  'examples',
])

/**
 * Convert a JSON Schema fragment into the Gemini Schema shape (uppercase
 * `type`, no unsupported metadata fields).
 * @param schema - a JSON Schema object or primitive.
 * @returns the Gemini-compatible schema.
 */
export function toGeminiSchema(schema: unknown): unknown {
  if (schema === null || typeof schema !== 'object' || Array.isArray(schema)) return schema
  const input = schema as Record<string, unknown>
  const result: Record<string, unknown> = {}
  const propertyNames = new Set<string>()
  if (input.properties !== undefined && typeof input.properties === 'object' && input.properties !== null) {
    for (const name of Object.keys(input.properties as Record<string, unknown>)) propertyNames.add(name)
  }
  for (const [key, value] of Object.entries(input)) {
    if (UNSUPPORTED_SCHEMA_FIELDS.has(key)) continue
    if (key === 'type' && typeof value === 'string') {
      result[key] = value.toUpperCase()
    } else if (key === 'properties' && typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const props: Record<string, unknown> = {}
      for (const [name, nested] of Object.entries(value as Record<string, unknown>)) {
        props[name] = toGeminiSchema(nested)
      }
      result[key] = props
    } else if (key === 'items') {
      result[key] = toGeminiSchema(value)
    } else if ((key === 'anyOf' || key === 'oneOf' || key === 'allOf') && Array.isArray(value)) {
      result[key] = value.map(item => toGeminiSchema(item))
    } else if (key === 'required' && Array.isArray(value) && propertyNames.size > 0) {
      const valid = value.filter((name): name is string => typeof name === 'string' && propertyNames.has(name))
      if (valid.length > 0) result[key] = valid
    } else {
      result[key] = value
    }
  }
  if (result.type === 'ARRAY' && result.items === undefined) result.items = { type: 'STRING' }
  return result
}

/** Flatten a tool result's content to plain text. */
function toolResultText(block: ToolResultBlock): string {
  return block.content.map(part => (part.type === 'text' ? part.text : '')).join('')
}

/** Parse a tool call's raw JSON arguments into an object. */
function parseToolInput(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return {}
  } catch {
    return {}
  }
}

/**
 * Convert harness messages into Gemini `contents`. System-role messages are
 * handled by {@link toGeminiSystem} and skipped here. Consecutive same-role
 * turns merge. Images must arrive pre-resolved.
 * @param messages - ordered conversation messages with resolved images.
 * @returns Gemini contents in conversation order.
 */
export function toGeminiContents(messages: readonly TranslatableMessage[]): GeminiContent[] {
  const out: GeminiContent[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    const role = message.role === 'assistant' ? 'model' : 'user'
    const parts: Record<string, unknown>[] = []
    for (const block of message.content) {
      switch (block.type) {
        case 'text':
          if (block.text.length > 0) parts.push({ text: block.text })
          break
        case 'tool-call':
          parts.push({
            functionCall: {
              name: block.name,
              args: parseToolInput(block.arguments),
              id: String(block.id),
            },
          })
          break
        case 'tool-result':
          parts.push({
            functionResponse: {
              name: '',
              id: String(block.toolCallId),
              response: { result: toolResultText(block) },
            },
          })
          break
        case 'image':
          if ('dataBase64' in block) {
            parts.push({ inlineData: { mimeType: block.mediaType, data: block.dataBase64 } })
          }
          break
        default:
          break
      }
    }
    if (parts.length === 0) continue
    const last = out[out.length - 1]
    if (last !== undefined && last.role === role) last.parts.push(...parts)
    else out.push({ role, parts })
  }
  return out
}

/**
 * Build the Gemini `systemInstruction` object from the explicit system
 * prompt and any system-role messages.
 * @param system - explicit system prompt, when set.
 * @param messages - conversation messages; their system-role text is appended.
 * @returns `{ parts }` or `undefined` when there is no system text.
 */
export function toGeminiSystem(
  system?: string,
  messages?: readonly TranslatableMessage[],
): { parts: { text: string }[] } | undefined {
  const parts: { text: string }[] = []
  if (system !== undefined && system.length > 0) parts.push({ text: system })
  for (const message of messages ?? []) {
    if (message.role !== 'system') continue
    for (const block of message.content) {
      if (block.type === 'text' && block.text.length > 0) parts.push({ text: block.text })
    }
  }
  return parts.length === 0 ? undefined : { parts }
}

/**
 * Map harness tool schemas to Gemini `functionDeclarations`.
 * @param tools - tool schemas from the request.
 * @returns a single tools entry wrapping the declarations.
 */
export function toGeminiTools(tools: readonly ToolSchema[]): Record<string, unknown>[] {
  return [{
    functionDeclarations: tools.map(tool => ({
      name: tool.name,
      description: tool.description,
      parameters: toGeminiSchema(tool.parameters),
    })),
  }]
}

/** The subset of Antigravity SSE payloads this translator reads. */
export interface GeminiStreamPayload {
  response?: {
    candidates?: Array<{
      content?: { parts?: GeminiStreamPart[] }
      finishReason?: string
    }>
    usageMetadata?: {
      promptTokenCount?: number
      candidatesTokenCount?: number
      thoughtsTokenCount?: number
    }
  }
  error?: { message?: string; status?: string; code?: number }
}

/** One Gemini response part. */
export interface GeminiStreamPart {
  text?: string
  thought?: boolean
  functionCall?: { name?: string; args?: Record<string, unknown>; id?: string }
}

/** One open harness block under assembly. */
interface OpenBlock {
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  callId: string
  name?: string
}

/** Assemble the final ContentBlock for one open block. */
function closeBlock(block: OpenBlock): ContentBlock {
  switch (block.kind) {
    case 'text':
      return { type: 'text', text: block.text }
    case 'reasoning':
      return { type: 'reasoning', text: block.text }
    case 'tool-call':
      return {
        type: 'tool-call',
        id: CallId(block.callId),
        name: block.name ?? '',
        arguments: block.text,
      }
  }
}

/**
 * Push-model Gemini / Antigravity SSE translator: feed each parsed payload
 * to {@link push} and collect the emitted harness StreamChunks.
 */
export class GeminiStreamTranslator {
  private nextIndex = 0
  private text: OpenBlock | undefined
  private reasoning: OpenBlock | undefined
  private tools = new Map<string, OpenBlock>()
  private pendingUsage: TokenUsage | undefined
  private stopReason: 'stop' | 'tool-calls' | 'max-tokens' = 'stop'
  private usageEmitted = false
  private sawAnyBlock = false
  /** Set once a finishReason produced the terminal finish chunk. */
  terminated = false

  private open(kind: OpenBlock['kind'], callId = '', name?: string): OpenBlock {
    const block: OpenBlock = {
      index: this.nextIndex++,
      kind,
      text: '',
      callId,
      ...name === undefined ? {} : { name },
    }
    this.sawAnyBlock = true
    return block
  }

  private emitUsage(chunks: StreamChunk[]): void {
    if (this.usageEmitted) return
    this.usageEmitted = true
    chunks.push({
      type: 'usage',
      usage: this.pendingUsage ?? { inputTokens: 0, outputTokens: 0 },
    })
  }

  private closeOpen(chunks: StreamChunk[]): void {
    if (this.reasoning !== undefined) {
      chunks.push({ type: 'block-end', index: this.reasoning.index, block: closeBlock(this.reasoning) })
      this.reasoning = undefined
    }
    if (this.text !== undefined) {
      chunks.push({ type: 'block-end', index: this.text.index, block: closeBlock(this.text) })
      this.text = undefined
    }
    for (const [key, block] of this.tools) {
      this.tools.delete(key)
      chunks.push({ type: 'block-end', index: block.index, block: closeBlock(block) })
    }
  }

  /**
   * Process one parsed Antigravity SSE payload.
   * @param payload - the parsed event object.
   * @returns the StreamChunks this event produced (possibly none).
   */
  push(payload: GeminiStreamPayload): StreamChunk[] {
    if (this.terminated) return []
    if (payload.error !== undefined) {
      throw new LlmError(payload.error.message ?? 'Antigravity reported an error', 'SERVER')
    }
    const chunks: StreamChunk[] = []
    const response = payload.response
    if (response === undefined) return chunks
    const usage = response.usageMetadata
    if (usage !== undefined) {
      this.pendingUsage = {
        inputTokens: usage.promptTokenCount ?? 0,
        outputTokens: (usage.candidatesTokenCount ?? 0) + (usage.thoughtsTokenCount ?? 0),
      }
    }
    const candidate = response.candidates?.[0]
    for (const part of candidate?.content?.parts ?? []) {
      if (part.thought === true && typeof part.text === 'string') {
        if (this.reasoning === undefined) {
          this.reasoning = this.open('reasoning')
          chunks.push({ type: 'block-start', index: this.reasoning.index, blockType: 'reasoning' })
        }
        this.reasoning.text += part.text
        chunks.push({ type: 'reasoning-delta', index: this.reasoning.index, text: part.text })
        continue
      }
      if (typeof part.text === 'string' && part.text.length > 0) {
        if (this.text === undefined) {
          this.text = this.open('text')
          chunks.push({ type: 'block-start', index: this.text.index, blockType: 'text' })
        }
        this.text.text += part.text
        chunks.push({ type: 'text-delta', index: this.text.index, text: part.text })
        continue
      }
      if (part.functionCall !== undefined) {
        const name = part.functionCall.name ?? ''
        const callId = part.functionCall.id ?? `call-${String(this.nextIndex)}`
        const key = part.functionCall.id ?? name
        let block = this.tools.get(key)
        if (block === undefined) {
          block = this.open('tool-call', callId, name)
          this.tools.set(key, block)
          chunks.push({ type: 'block-start', index: block.index, blockType: 'tool-call' })
          chunks.push({
            type: 'tool-call-delta',
            index: block.index,
            id: CallId(block.callId),
            ...name.length === 0 ? {} : { name },
            argumentsDelta: '',
          })
        }
        const args = part.functionCall.args === undefined ? '' : JSON.stringify(part.functionCall.args)
        if (args.length > 0 && args !== block.text) {
          const delta = args.slice(block.text.length)
          block.text = args
          if (delta.length > 0) {
            chunks.push({
              type: 'tool-call-delta',
              index: block.index,
              id: CallId(block.callId),
              ...block.name === undefined ? {} : { name: block.name },
              argumentsDelta: delta,
            })
          }
        }
        this.stopReason = 'tool-calls'
      }
    }
    const finish = candidate?.finishReason
    if (finish === undefined || finish.length === 0) return chunks
    this.terminated = true
    if (finish === 'MAX_TOKENS') this.stopReason = 'max-tokens'
    else if (this.tools.size > 0 || finish === 'OTHER') {
      if (this.tools.size > 0) this.stopReason = 'tool-calls'
    } else {
      this.stopReason = 'stop'
    }
    this.closeOpen(chunks)
    this.emitUsage(chunks)
    if (this.stopReason === 'stop' && !this.sawAnyBlock) {
      chunks.push({
        type: 'finish',
        reason: {
          kind: 'error',
          failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE },
        },
      })
    } else {
      chunks.push({ type: 'finish', reason: { kind: this.stopReason } })
    }
    return chunks
  }
}

/**
 * Consume an Antigravity SSE byte stream and yield harness StreamChunks.
 * @param stream - raw response body.
 * @param onActivity - transport-activity callback for the idle watchdog.
 * @returns the chunk stream; throws when the stream ends before a finish.
 */
export async function* streamGemini(
  stream: ReadableStream<Uint8Array>,
  onActivity?: () => void,
): AsyncGenerator<StreamChunk> {
  const translator = new GeminiStreamTranslator()
  for await (const sseEvent of parseSse(stream, onActivity)) {
    if (sseEvent.data === '[DONE]') {
      if (!translator.terminated) {
        yield* translator.push({ response: { candidates: [{ finishReason: 'STOP' }] } })
      }
      return
    }
    let payload: GeminiStreamPayload
    try {
      payload = JSON.parse(sseEvent.data) as GeminiStreamPayload
    } catch {
      throw new LlmError(`malformed SSE payload: ${sseEvent.data.slice(0, 120)}`, 'MALFORMED_RESPONSE')
    }
    yield* translator.push(payload)
    if (translator.terminated) return
  }
  throw new LlmError('Antigravity SSE stream ended before a finishReason', 'STREAM_CLOSED')
}
