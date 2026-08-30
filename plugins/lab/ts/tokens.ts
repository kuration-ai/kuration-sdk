import { Token } from '@lumino/coreutils';

import { KurationClient } from './api/client';
import { RunRegistryModel } from './models/runRegistry';

export const IKurationClient = new Token<KurationClient>(
  'kuration-sdk-lab:client',
  'HTTP/WebSocket client for the kuration server extension.'
);

export const IKurationRunRegistry = new Token<RunRegistryModel>(
  'kuration-sdk-lab:run-registry',
  'Shared model of all kuration runs and their live progress.'
);

export namespace CommandIDs {
  export const newJob = 'kuration:new-job';
  export const openJobPanel = 'kuration:open-job-panel';
  export const runJob = 'kuration:run-job';
  export const openRun = 'kuration:open-run';
  export const cancelRun = 'kuration:cancel-run';
  export const inspectDataset = 'kuration:inspect-dataset';
}
