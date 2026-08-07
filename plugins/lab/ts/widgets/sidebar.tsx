import { ReactWidget, UseSignal } from '@jupyterlab/ui-components';
import * as React from 'react';

import { StateBadge } from '../components/common';
import { RunModel } from '../models/runModel';
import { RunRegistryModel } from '../models/runRegistry';

function RunRow(props: {
  model: RunModel;
  onOpen: (runId: string) => void;
}): JSX.Element {
  const info = props.model.info;
  const lastPhase = [...props.model.phases.values()].pop();
  return (
    <li className="kur-run-row" onClick={() => props.onOpen(info.run_id)}>
      <StateBadge state={info.state} />
      <div className="kur-run-row-body">
        <div className="kur-run-row-id">{info.run_id}</div>
        <div className="kur-muted">
          {info.kind}
          {lastPhase && info.state === 'running'
            ? ` · ${lastPhase.phase} ${lastPhase.current}${
                lastPhase.total ? '/' + lastPhase.total : ''
              }`
            : ''}
        </div>
      </div>
    </li>
  );
}

function SidebarView(props: {
  registry: RunRegistryModel;
  onOpenRun: (runId: string) => void;
  onNewJob: () => void;
  onInspect: () => void;
}): JSX.Element {
  const runs = props.registry.runs;
  return (
    <div className="kur-sidebar">
      <div className="kur-sidebar-actions">
        <button className="jp-Button" onClick={props.onNewJob}>
          New Job
        </button>
        <button className="jp-Button" onClick={props.onInspect}>
          Inspect Dataset…
        </button>
        <button className="jp-Button" onClick={() => void props.registry.refresh()}>
          ⟳
        </button>
      </div>
      <h3 className="kur-sidebar-heading">Runs</h3>
      {runs.length ? (
        <ul className="kur-run-list">
          {runs.map(model => (
            <RunRow key={model.info.run_id} model={model} onOpen={props.onOpenRun} />
          ))}
        </ul>
      ) : (
        <div className="kur-muted kur-sidebar-empty">
          No runs yet. Create a job and press Run.
        </div>
      )}
    </div>
  );
}

export class KurationSidebar extends ReactWidget {
  constructor(options: {
    registry: RunRegistryModel;
    onOpenRun: (runId: string) => void;
    onNewJob: () => void;
    onInspect: () => void;
  }) {
    super();
    this._options = options;
    this.id = 'kuration-sidebar';
    this.title.caption = 'Kuration';
    this.title.iconClass = 'jp-SideBar-tabIcon jp-SpreadsheetIcon';
    this.addClass('kur-widget');
  }

  protected render(): JSX.Element {
    return (
      <UseSignal signal={this._options.registry.runsChanged}>
        {() => <SidebarView {...this._options} />}
      </UseSignal>
    );
  }

  private readonly _options: {
    registry: RunRegistryModel;
    onOpenRun: (runId: string) => void;
    onNewJob: () => void;
    onInspect: () => void;
  };
}
