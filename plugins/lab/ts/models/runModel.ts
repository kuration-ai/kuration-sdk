import { ISignal, Signal } from '@lumino/signaling';

import { KurationClient } from '../api/client';
import { ProgressStream } from '../api/progress';
import { IProgressEvent, IRunInfo, TERMINAL_STATES } from '../api/types';

export interface IPhaseProgress {
  phase: string;
  current: number;
  total: number | null;
  status: string;
  source: string;
  detail: string;
  done: boolean;
}

const EVENT_LOG_LIMIT = 200;

/**
 * One run's live state: info snapshot + per-phase progress folded from the event stream.
 * Sidebar rows and the RunMonitor widget bind to the same model, so progress is consistent
 * everywhere.
 */
export class RunModel {
  constructor(client: KurationClient, info: IRunInfo) {
    this._client = client;
    this._info = info;
  }

  get info(): IRunInfo {
    return this._info;
  }

  get phases(): ReadonlyMap<string, IPhaseProgress> {
    return this._phases;
  }

  get recentEvents(): readonly IProgressEvent[] {
    return this._events;
  }

  get changed(): ISignal<this, void> {
    return this._changed;
  }

  get isTerminal(): boolean {
    return TERMINAL_STATES.has(this._info.state);
  }

  get streamDegraded(): boolean {
    return this._stream?.degraded ?? false;
  }

  /** Start (or restart) live tracking; snapshot first, then stream deltas. */
  track(): void {
    if (this.isTerminal || this._stream) {
      return;
    }
    const stream = new ProgressStream(this._client, this._info.run_id, 0);
    this._stream = stream;
    stream.eventReceived.connect((_, event) => this._apply(event));
    stream.terminated.connect(() => {
      void this.refresh();
      this._disposeStream();
    });
    stream.start();
  }

  async refresh(): Promise<void> {
    try {
      this._info = await this._client.getRun(this._info.run_id);
    } catch {
      return; // purged or unreachable; keep the last snapshot
    }
    this._changed.emit();
  }

  update(info: IRunInfo): void {
    this._info = info;
    this._changed.emit();
  }

  cancel(): void {
    if (this._stream) {
      this._stream.cancelRun();
    } else {
      void this._client.cancelRun(this._info.run_id).then(info => this.update(info));
    }
  }

  dispose(): void {
    this._disposeStream();
    Signal.clearData(this);
  }

  private _apply(event: IProgressEvent): void {
    this._events.push(event);
    if (this._events.length > EVENT_LOG_LIMIT) {
      this._events.splice(0, this._events.length - EVENT_LOG_LIMIT);
    }
    if (event.phase && event.phase !== 'lifecycle') {
      const previous = this._phases.get(event.phase);
      this._phases.set(event.phase, {
        phase: event.phase,
        current: event.current ?? previous?.current ?? 0,
        total: event.total !== undefined ? event.total : (previous?.total ?? null),
        status: event.status ?? previous?.status ?? '',
        source: event.source ?? previous?.source ?? '',
        detail: event.detail ?? previous?.detail ?? '',
        done: event.event === 'complete'
      });
    }
    this._changed.emit();
  }

  private _disposeStream(): void {
    this._stream?.dispose();
    this._stream = null;
  }

  private readonly _client: KurationClient;
  private _info: IRunInfo;
  private _stream: ProgressStream | null = null;
  private _phases = new Map<string, IPhaseProgress>();
  private _events: IProgressEvent[] = [];
  private _changed = new Signal<this, void>(this);
}
