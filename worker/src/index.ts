import { RolloutWorker, optionsFromEnvironment } from './worker.js'

const worker = new RolloutWorker(optionsFromEnvironment())

async function shutdown(signal: string): Promise<void> {
  console.log(`received ${signal}; stopping worker`)
  await worker.stop()
  process.exit(0)
}

process.on('SIGINT', () => void shutdown('SIGINT'))
process.on('SIGTERM', () => void shutdown('SIGTERM'))

worker.start().catch(error => {
  console.error(error)
  process.exitCode = 1
})
