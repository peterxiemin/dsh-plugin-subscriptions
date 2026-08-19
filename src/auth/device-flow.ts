/**
 * RFC 8628 device-authorization-grant engine. One attempt per provider at a
 * time: start the device request, hand the verification URL to the UI, and
 * poll until the user approves, the attempt is cancelled, or it expires.
 */

/** One device-authorization response (subset of RFC 8628). */
export interface DeviceAuthorization {
  deviceCode: string
  userCode: string
  /** Fallback URL shown to the user if the complete URL is unusable. */
  verificationUri: string
  /** Prefill URL opened in the browser (includes the user code). */
  verificationUriComplete: string
  intervalSeconds: number
  expiresInSeconds: number
}

/** One in-flight device-code login. */
export interface DeviceAttempt<S> {
  /** URL to open in the user's browser (`verification_uri_complete`). */
  readonly authorizeUrl: string
  /** Short user code, shown as a fallback when the browser flow fails. */
  readonly userCode: string
  /**
   * Wait until polling yields a session.
   * @returns the persisted session; rejects on timeout, denial, or cancel.
   */
  waitSession(): Promise<S>
  /** Abort the attempt and stop polling. */
  cancel(): void
}

/**
 * Own the set of in-flight device-code attempts, keyed by provider. One
 * attempt per provider at a time; an attempt removes itself when it settles.
 */
export class DeviceFlowManager {
  private attempts = new Map<string, DeviceAttempt<unknown>>()

  /**
   * Whether a device-code login is running for one provider.
   * @param provider - the provider route.
   * @returns true while an attempt is waiting for approval.
   */
  isBusy(provider: string): boolean {
    return this.attempts.has(provider)
  }

  /**
   * The pending attempt for one provider, when any.
   * @param provider - the provider route.
   * @returns the in-flight attempt, or `undefined`.
   */
  pending(provider: string): DeviceAttempt<unknown> | undefined {
    return this.attempts.get(provider)
  }

  /**
   * Start a device-code login: request the user/device codes, then poll in
   * the background until the grant completes.
   * @param provider - the provider route (one attempt at a time).
   * @param authorize - perform the device-authorization request.
   * @param poll - poll the token endpoint until the user approves.
   * @returns the live attempt; its `waitSession()` settles the login.
   * @throws when an attempt is already running.
   */
  async start<S>(
    provider: string,
    authorize: () => Promise<DeviceAuthorization>,
    poll: (device: DeviceAuthorization, signal: AbortSignal) => Promise<S>,
  ): Promise<DeviceAttempt<S>> {
    if (this.attempts.has(provider)) {
      throw new Error(`a ${provider} login attempt is already in progress`)
    }
    const device = await authorize()
    const controller = new AbortController()
    let settled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let resolveSession!: (session: S) => void
    let rejectSession!: (error: Error) => void
    const sessionPromise = new Promise<S>((resolve, reject) => {
      resolveSession = resolve
      rejectSession = reject
    })

    const settle = (error?: Error, session?: S): void => {
      if (settled) return
      settled = true
      if (timer !== undefined) clearTimeout(timer)
      if (!controller.signal.aborted) controller.abort(error ?? new Error('login settled'))
      this.attempts.delete(provider)
      if (error !== undefined) rejectSession(error)
      else if (session !== undefined) resolveSession(session)
    }

    timer = setTimeout(() => {
      settle(new Error(`login timed out after ${String(device.expiresInSeconds)}s`))
    }, device.expiresInSeconds * 1000)
    timer.unref()

    const attempt: DeviceAttempt<S> = {
      authorizeUrl: device.verificationUriComplete,
      userCode: device.userCode,
      waitSession: () => sessionPromise,
      cancel() {
        settle(new Error('login cancelled'))
      },
    }
    this.attempts.set(provider, attempt as DeviceAttempt<unknown>)

    void poll(device, controller.signal).then(
      session => settle(undefined, session),
      (error: unknown) => {
        if (settled) return
        settle(error instanceof Error ? error : new Error(String(error)))
      },
    )
    return attempt
  }
}
