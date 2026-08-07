import { ISignal, Signal } from '@lumino/signaling';

import { KurationClient } from './client';
import { IProgressEvent, RunState, TERMINAL_STATES } from './types';

const BACKOFF_BASE_MS = 1000;
const BACKOFF_CAP_MS = 30_000;
const POLL_FALLBACK_AFTER = 5;
const POLL_INTERVAL_MS = 2000;

/**
 * Live event stream for one run: WebSocket with replay-from-seq, exponential-backoff
 * reconnect, and a polling fallback after repeated failures. Every (re)connect resumes
 * from the last seen seq, so a dropped connection never loses events.
 */
export class ProgressStream {
  constructor(client: KurationClient, runId: string, since = 0) {
    this._client = client;
    this._runId = runId;
    this._since = since;
  }

  get eventReceived(): ISignal<this, IProgressEvent> {
    return this._eventReceived;
  }

  get terminated(): ISignal<this, RunState> {
    return this._terminated;
  }

  get degraded(): boolean {
    return this._degraded;
  }

  start(): void {
    if (this._disposed || this._socket || this._pollTimer !== null) {
      return;
    }
    this._connect();
  }

  dispose(): void {
    this._disposed = true;
    this._closeSocket();
    if (this._pollTimer !== null) {
      window.clearTimeout(this._pollTimer);
      this._pollTimer = null;
    }
    Signal.clearData(this);
  }

  cancelRun(): void {
    if (this._socket && this._socket.readyState === WebSocket.OPEN) {
      this._socket.send(JSON.stringify({ op: 'cancel' }));
    } else {
      void this._client.cancelRun(this._runId);
    }
  }

  private _connect(): void {
    if (this._disposed) {
      return;
    }
    const socket = new WebSocket(this._client.wsUrl(this._runId, this._since));
    this._socket = socket;
    socket.onmessage = message => {
      this._failures = 0;
      let event: IProgressEvent;
      try {
        event = JSON.parse(message.data as string) as IProgressEvent;
      } catch {
        return;
      }
      this._handle(event);
    };
    socket.onclose = closeEvent => {
      this._socket = null;
      if (this._disposed || this._terminal) {
        return;
      }
      if (closeEvent.code === 1000) {
        return; // server finished the stream cleanly
      }
      this._failures += 1;
      if (this._failures >= POLL_FALLBACK_AFTER) {
        this._degraded = true;
        this._schedulePoll();
      } else {
        const delay = Math.min(
          BACKOFF_CAP_MS,
          BACKOFF_BASE_MS * 2 ** (this._failures - 1)
        );
        window.setTimeout(() => this._connect(), delay + Math.random() * 500);
      }
    };
    socket.onerror = () => {
      // onclose follows and handles retry
    };
  }

  private _handle(event: IProgressEvent): void {
    if (typeof event.seq === 'number' && event.seq > this._since) {
      this._since = event.seq;
    }
    if (event.phase === 'lifecycle' && event.event === 'state') {
      this._terminal = true;
      this._terminated.emit((event.state ?? 'succeeded') as RunState);
      this._closeSocket();
      return;
    }
    this._eventReceived.emit(event);
  }

  private _schedulePoll(): void {
    if (this._disposed || this._terminal) {
      return;
    }
    this._pollTimer = window.setTimeout(() => {
      void this._poll();
    }, POLL_INTERVAL_MS);
  }

  private async _poll(): Promise<void> {
    this._pollTimer = null;
    try {
      const page = await this._client.runEvents(this._runId, this._since);
      for (const event of page.events) {
        this._handle(event);
      }
      if (TERMINAL_STATES.has(page.state) && !this._terminal) {
        this._terminal = true;
        this._terminated.emit(page.state);
        return;
      }
    } catch {
      // transient; keep polling
    }
    this._schedulePoll();
  }

  private _closeSocket(): void {
    if (this._socket) {
      this._socket.onclose = null;
      this._socket.onmessage = null;
      this._socket.close();
      this._socket = null;
    }
  }

  private readonly _client: KurationClient;
  private readonly _runId: string;
  private _since: number;
  private _socket: WebSocket | null = null;
  private _pollTimer: number | null = null;
  private _failures = 0;
  private _degraded = false;
  private _terminal = false;
  private _disposed = false;
  private _eventReceived = new Signal<this, IProgressEvent>(this);
  private _terminated = new Signal<this, RunState>(this);
}
