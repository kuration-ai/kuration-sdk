import { ISignal, Signal } from '@lumino/signaling';

import { KurationClient } from '../api/client';
import { IRunInfo, JobDocument } from '../api/types';
import { RunModel } from './runModel';

const REFRESH_INTERVAL_MS = 10_000;

/** All known runs, kept fresh by a light poll; live runs additionally stream events. */
export class RunRegistryModel {
  constructor(client: KurationClient) {
    this.client = client;
  }

  readonly client: KurationClient;

  get runsChanged(): ISignal<this, void> {
    return this._runsChanged;
  }

  get runs(): readonly RunModel[] {
    return [...this._models.values()].sort((a, b) =>
      (b.info.created ?? '').localeCompare(a.info.created ?? '')
    );
  }

  get(runId: string): RunModel | undefined {
    return this._models.get(runId);
  }

  startPolling(): void {
    if (this._timer !== null) {
      return;
    }
    const tick = async (): Promise<void> => {
      await this.refresh();
      this._timer = window.setTimeout(() => void tick(), REFRESH_INTERVAL_MS);
    };
    void tick();
  }

  async refresh(): Promise<void> {
    let runs: IRunInfo[];
    try {
      runs = (await this.client.listRuns()).runs;
    } catch {
      return;
    }
    let changed = false;
    for (const info of runs) {
      const existing = this._models.get(info.run_id);
      if (existing) {
        if (
          existing.info.state !== info.state ||
          existing.info.updated !== info.updated
        ) {
          existing.update(info);
          changed = true;
        }
      } else {
        const model = new RunModel(this.client, info);
        model.track();
        model.changed.connect(() => this._runsChanged.emit());
        this._models.set(info.run_id, model);
        changed = true;
      }
    }
    const known = new Set(runs.map(run => run.run_id));
    for (const [runId, model] of this._models) {
      if (!known.has(runId)) {
        model.dispose();
        this._models.delete(runId);
        changed = true;
      }
    }
    if (changed) {
      this._runsChanged.emit();
    }
  }

  async submit(
    job: JobDocument,
    options: { jobDir?: string; resume?: boolean; force?: boolean } = {}
  ): Promise<RunModel> {
    const info = await this.client.submitRun(job, options);
    await this.refresh();
    const model = this._models.get(info.run_id);
    if (!model) {
      throw new Error(`submitted run ${info.run_id} did not appear in the registry`);
    }
    return model;
  }

  dispose(): void {
    if (this._timer !== null) {
      window.clearTimeout(this._timer);
      this._timer = null;
    }
    for (const model of this._models.values()) {
      model.dispose();
    }
    this._models.clear();
    Signal.clearData(this);
  }

  private _models = new Map<string, RunModel>();
  private _timer: number | null = null;
  private _runsChanged = new Signal<this, void>(this);
}
