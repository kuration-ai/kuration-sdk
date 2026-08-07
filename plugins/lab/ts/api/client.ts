import { URLExt } from '@jupyterlab/coreutils';
import { ServerConnection } from '@jupyterlab/services';

import {
  IEventsPage,
  IErrorEnvelope,
  IRunInfo,
  IShardsResponse,
  IStatsResponse,
  IVersionInfo,
  JobDocument
} from './types';

const API_NAMESPACE = 'kuration/v1';

/** Error carrying the server's envelope (type, message, actionable hint). */
export class KurationApiError extends Error {
  readonly type: string;
  readonly hint: string | null;
  readonly status: number;

  constructor(status: number, envelope: IErrorEnvelope['error']) {
    super(envelope.message);
    this.name = 'KurationApiError';
    this.type = envelope.type;
    this.hint = envelope.hint;
    this.status = status;
  }
}

export class KurationClient {
  readonly serverSettings: ServerConnection.ISettings;

  constructor(settings?: ServerConnection.ISettings) {
    this.serverSettings = settings ?? ServerConnection.makeSettings();
  }

  private url(...parts: string[]): string {
    return URLExt.join(this.serverSettings.baseUrl, API_NAMESPACE, ...parts);
  }

  /** WebSocket URL for a run's event stream. */
  wsUrl(runId: string, since: number): string {
    const base = URLExt.join(
      this.serverSettings.wsUrl,
      API_NAMESPACE,
      'ws',
      'runs',
      runId
    );
    const token = this.serverSettings.token;
    const query = new URLSearchParams({ since: String(since) });
    if (token) {
      query.set('token', token);
    }
    return `${base}?${query.toString()}`;
  }

  private async raw(
    parts: string[],
    init: RequestInit = {}
  ): Promise<Response> {
    const response = await ServerConnection.makeRequest(
      this.url(...parts),
      init,
      this.serverSettings
    );
    if (!response.ok) {
      let envelope: IErrorEnvelope['error'] | null = null;
      try {
        envelope = ((await response.json()) as IErrorEnvelope).error ?? null;
      } catch {
        // non-JSON error body
      }
      throw new KurationApiError(
        response.status,
        envelope ?? {
          type: 'HTTPError',
          message: `${response.status} ${response.statusText}`,
          hint: null
        }
      );
    }
    return response;
  }

  private async json<T>(parts: string[], init: RequestInit = {}): Promise<T> {
    const response = await this.raw(parts, init);
    return (await response.json()) as T;
  }

  private post<T>(parts: string[], body: unknown): Promise<T> {
    return this.json<T>(parts, { method: 'POST', body: JSON.stringify(body) });
  }

  // ---- version / authoring ------------------------------------------------

  version(): Promise<IVersionInfo> {
    return this.json<IVersionInfo>(['version']);
  }

  jobSchema(): Promise<Record<string, any>> {
    return this.json<Record<string, any>>(['jobs', 'schema']);
  }

  infer(body: Record<string, any>): Promise<Record<string, any>> {
    return this.post(['infer'], body);
  }

  buildJob(body: Record<string, any>): Promise<{ job: JobDocument }> {
    return this.post(['jobs', 'build'], body);
  }

  validateJob(job: JobDocument, jobDir?: string): Promise<Record<string, any>> {
    return this.post(['jobs', 'validate'], { job, job_dir: jobDir });
  }

  migrateJob(job: JobDocument, jobDir?: string): Promise<{ job: JobDocument }> {
    return this.post(['jobs', 'migrate'], { job, job_dir: jobDir });
  }

  explainJob(job: JobDocument, jobDir?: string): Promise<Record<string, any>> {
    return this.post(['jobs', 'explain'], { job, job_dir: jobDir });
  }

  preview(job: JobDocument, jobDir?: string): Promise<Record<string, any>> {
    return this.post(['preview'], { job, job_dir: jobDir });
  }

  // ---- runs -----------------------------------------------------------------

  submitRun(
    job: JobDocument,
    options: { jobDir?: string; resume?: boolean; force?: boolean } = {}
  ): Promise<IRunInfo> {
    return this.post(['runs'], {
      job,
      job_dir: options.jobDir,
      resume: options.resume,
      force: options.force
    });
  }

  listRuns(): Promise<{ runs: IRunInfo[] }> {
    return this.json<{ runs: IRunInfo[] }>(['runs']);
  }

  getRun(runId: string): Promise<IRunInfo> {
    return this.json<IRunInfo>(['runs', runId]);
  }

  cancelRun(runId: string): Promise<IRunInfo> {
    return this.json<IRunInfo>(['runs', runId], { method: 'DELETE' });
  }

  purgeRun(runId: string): Promise<{ purged: boolean }> {
    return this.json<{ purged: boolean }>(['runs', `${runId}?purge=true`], {
      method: 'DELETE'
    });
  }

  runEvents(runId: string, since = 0, limit = 1000): Promise<IEventsPage> {
    return this.json<IEventsPage>([
      'runs',
      runId,
      `events?since=${since}&limit=${limit}`
    ]);
  }

  runResult(runId: string): Promise<Record<string, any>> {
    return this.json<Record<string, any>>(['runs', runId, 'result']);
  }

  // ---- inspection -----------------------------------------------------------

  inspect(dataset: string, verify = false): Promise<Record<string, any>> {
    return this.post(['inspect'], { dataset, verify });
  }

  compare(left: string, right: string): Promise<Record<string, any>> {
    return this.post(['compare'], { left, right });
  }

  stats(subject: string, options: Record<string, any> = {}): Promise<IStatsResponse> {
    return this.post(['stats'], { subject, ...options });
  }

  statsExamples(
    subject: string,
    options: Record<string, any> = {}
  ): Promise<Record<string, any>> {
    return this.post(['stats', 'examples'], { subject, ...options });
  }

  async statsHtml(subject: string, title?: string): Promise<string> {
    const response = await this.raw(['stats', 'html'], {
      method: 'POST',
      body: JSON.stringify({ subject, title })
    });
    return response.text();
  }

  /** Returns an object URL for the rendered PNG; caller revokes it when done. */
  async statsPlot(
    subject: string,
    options: { kind?: string; metric?: string } = {}
  ): Promise<string> {
    const response = await this.raw(['stats', 'plot'], {
      method: 'POST',
      body: JSON.stringify({ subject, ...options })
    });
    return URL.createObjectURL(await response.blob());
  }

  shards(dataset: string, split?: string): Promise<IShardsResponse> {
    return this.post(['shards'], { dataset, split });
  }

  async shardArrays(
    dataset: string,
    entry: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<Blob> {
    const response = await this.raw(['shards', 'arrays'], {
      method: 'POST',
      body: JSON.stringify({ dataset, entry, ...options })
    });
    return response.blob();
  }

  frames(
    job: JobDocument,
    options: Record<string, any> = {}
  ): Promise<{ media: string; source: string; frames: string[] }> {
    return this.post(['frames'], { job, ...options });
  }
}
