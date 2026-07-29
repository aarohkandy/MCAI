import fs from 'node:fs'
import path from 'node:path'
import type { ActionV1, ObservationV1 } from './contracts.js'

/**
 * Records (observation, action) pairs as whole-match JSONL for behavior cloning.
 *
 * Why this exists: self-play from a uniform-random policy almost never discovers the long action
 * chains this game requires (approach → face → attack; or place obsidian → place crystal → hit it).
 * The scripted teacher in scripted-policy.ts already performs those chains competently, but it is
 * discarded the moment the trainer connects. Recording it produces demonstrations the trainer can
 * behavior-clone from (`mcai-trainer serve --imitation-data <file>`), which bootstraps the policy
 * past the exploration barrier for free.
 *
 * The emitted schema matches trainer/combat_ai/imitation.py load_demonstrations():
 * one JSON object per line with `match_id`, `observation`, and `action`.
 */
export class DemoRecorder {
  private readonly stream: fs.WriteStream
  private recorded = 0

  constructor(private readonly destination: string) {
    fs.mkdirSync(path.dirname(path.resolve(destination)), { recursive: true })
    // Append so repeated runs accumulate demonstrations into one corpus.
    this.stream = fs.createWriteStream(destination, { flags: 'a' })
  }

  record(observation: ObservationV1, action: ActionV1): void {
    // episode_id groups rows into whole matches, which is the unit split_matches() splits on.
    const line = JSON.stringify({
      match_id: observation.match.episode_id,
      observation,
      action
    })
    this.stream.write(`${line}\n`)
    this.recorded += 1
  }

  get count(): number {
    return this.recorded
  }

  async close(): Promise<void> {
    await new Promise<void>(resolve => this.stream.end(resolve))
  }
}
