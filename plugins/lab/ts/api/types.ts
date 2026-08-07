/** Payload shapes for the /kuration/v1 API (SDK TypedResult dicts pass through as-is). */

export interface IVersionInfo {
  api: number;
  lab: string;
  sdk: string;
  job_schema_version: string;
}

export type RunState =
  | 'pending'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'orphaned';

export const TERMINAL_STATES: ReadonlySet<RunState> = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'orphaned'
]);

export interface IRunExit {
  ok: boolean;
  cancelled?: boolean;
  error_type?: string;
  message?: string;
  hint?: string | null;
}

export interface IRunInfo {
  run_id: string;
  kind: string;
  out: string | null;
  created: string | null;
  submitter: string | null;
  state: RunState;
  updated: string | null;
  exit: IRunExit | null;
  pid: number | null;
  has_result: boolean;
  resume_available: boolean;
  last_event: IProgressEvent | null;
}

/** SDK progress event stamped with seq/run_id/ts by the worker. */
export interface IProgressEvent {
  seq: number;
  run_id: string;
  ts: string;
  phase: string;
  event: string;
  current?: number;
  total?: number | null;
  source?: string;
  status?: string;
  detail?: string;
  state?: RunState; // synthetic lifecycle "state" message
  exit?: IRunExit | null;
}

export interface IEventsPage {
  events: IProgressEvent[];
  next: number;
  state: RunState;
}

export interface IErrorEnvelope {
  error: { type: string; message: string; hint: string | null };
}

export interface IStatsResponse {
  report: Record<string, any>;
  capabilities: {
    plots?: string[];
    default_plot?: string;
    example_modes?: string[];
    comparisons?: string[];
    media_examples?: boolean;
    [key: string]: any;
  };
}

export interface IShardEntry {
  path: string;
  source_id: string;
  entity_id?: string;
  split?: string;
  track_id?: string;
  shard_index?: number;
  n_steps?: number;
  padded?: boolean;
  size_bytes?: number;
  sha256?: string;
  [key: string]: any;
}

export interface IShardsResponse {
  entries: IShardEntry[];
  container: string;
}

export type JobDocument = Record<string, any>;
