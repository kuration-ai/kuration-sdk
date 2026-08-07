import { ReactWidget, UseSignal } from '@jupyterlab/ui-components';
import * as React from 'react';

import { KurationClient } from '../api/client';
import { IProgressEvent } from '../api/types';
import { ErrorBox, JsonTree, KeyValueTable, StateBadge, useAsync } from '../components/common';
import { IPhaseProgress, RunModel } from '../models/runModel';

function PhaseBar(props: { progress: IPhaseProgress }): JSX.Element {
  const { progress } = props;
  const { current, total } = progress;
  const fraction =
    total && total > 0 ? Math.min(1, current / total) : progress.done ? 1 : 0;
  const label = total ? `${current}/${total}` : `${current || ''}`;
  return (
    <div className="kur-phase">
      <div className="kur-phase-header">
        <span className="kur-phase-name">{progress.phase}</span>
        <span className="kur-phase-count">
          {label} {progress.status ? `(${progress.status})` : ''}
        </span>
      </div>
      <div className="kur-progressbar">
        <div
          className={`kur-progressbar-fill${progress.done ? ' kur-done' : ''}`}
          style={{ width: `${Math.round(fraction * 100)}%` }}
        />
      </div>
      <div className="kur-phase-detail">
        {progress.source} {progress.detail}
      </div>
    </div>
  );
}

function EventLog(props: { events: readonly IProgressEvent[] }): JSX.Element {
  return (
    <details className="kur-eventlog">
      <summary>event log ({props.events.length})</summary>
      <pre>
        {props.events
          .map(
            event =>
              `${event.seq}\t${event.phase}/${event.event}\t` +
              `${event.current ?? ''}${event.total ? '/' + event.total : ''}\t` +
              `${event.source ?? ''} ${event.detail ?? ''}`
          )
          .join('\n')}
      </pre>
    </details>
  );
}

function RunResult(props: { client: KurationClient; runId: string }): JSX.Element {
  const { data, error, loading } = useAsync(
    () => props.client.runResult(props.runId),
    [props.runId]
  );
  if (loading) {
    return <div>loading result…</div>;
  }
  if (error) {
    return <ErrorBox error={error} />;
  }
  return <JsonTree value={data} label="result" />;
}

function RunView(props: { model: RunModel; client: KurationClient }): JSX.Element {
  const { model, client } = props;
  const info = model.info;
  const phases = [...model.phases.values()];
  return (
    <div className="kur-panel kur-run-monitor">
      <div className="kur-toolbar">
        <StateBadge state={info.state} />
        <span className="kur-title">{info.run_id}</span>
        {model.streamDegraded ? (
          <span className="kur-badge kur-badge-degraded">live updates degraded</span>
        ) : null}
        <span className="kur-spacer" />
        {!model.isTerminal ? (
          <button className="jp-Button" onClick={() => model.cancel()}>
            Cancel
          </button>
        ) : null}
        <button className="jp-Button" onClick={() => void model.refresh()}>
          Refresh
        </button>
      </div>
      <KeyValueTable
        rows={[
          ['kind', info.kind],
          ['output', info.out ?? '—'],
          ['created', info.created ?? '—'],
          ['updated', info.updated ?? '—'],
          ['resume available', String(info.resume_available)]
        ]}
      />
      {info.exit && !info.exit.ok && !info.exit.cancelled ? (
        <div className="kur-error">
          <strong>
            {info.exit.error_type ?? 'error'}: {info.exit.message ?? ''}
          </strong>
          {info.exit.hint ? (
            <div className="kur-error-hint">hint: {info.exit.hint}</div>
          ) : null}
        </div>
      ) : null}
      <div className="kur-phases">
        {phases.length ? (
          phases.map(progress => <PhaseBar key={progress.phase} progress={progress} />)
        ) : (
          <div className="kur-muted">
            {model.isTerminal
              ? 'no live events (finished earlier or before this session)'
              : 'waiting for events…'}
          </div>
        )}
      </div>
      <EventLog events={model.recentEvents} />
      {info.state === 'succeeded' && info.has_result ? (
        <RunResult client={client} runId={info.run_id} />
      ) : null}
    </div>
  );
}

export class RunMonitorWidget extends ReactWidget {
  constructor(model: RunModel, client: KurationClient) {
    super();
    this._model = model;
    this._client = client;
    this.addClass('kur-widget');
    this.title.label = `Run ${model.info.run_id}`;
    this.title.iconClass = 'jp-MaterialIcon jp-RunIcon';
    this.title.closable = true;
  }

  protected render(): JSX.Element {
    return (
      <UseSignal signal={this._model.changed}>
        {() => <RunView model={this._model} client={this._client} />}
      </UseSignal>
    );
  }

  private readonly _model: RunModel;
  private readonly _client: KurationClient;
}
