import net from 'node:net'
import { EventEmitter } from 'node:events'

type PendingRequest = {
  resolve: (value: Record<string, unknown>) => void
  reject: (error: Error) => void
  timeout: NodeJS.Timeout
}

export type ArenaEvent = {
  type: 'event'
  event: string
  arena_id?: string
  agent_id?: string
  payload?: Record<string, unknown>
}

export class ArenaClient extends EventEmitter {
  private socket: net.Socket | null = null
  private buffer = ''
  private sequence = 0
  private pending = new Map<number, PendingRequest>()

  constructor(private readonly host = '127.0.0.1', private readonly port = 8765) {
    super()
  }

  async connect(): Promise<void> {
    if (this.socket && !this.socket.destroyed) return
    this.socket = net.createConnection({ host: this.host, port: this.port })
    this.socket.setEncoding('utf8')
    this.socket.on('data', data => this.onData(String(data)))
    this.socket.on('error', error => this.emit('error', error))
    this.socket.on('close', () => {
      this.socket = null
      this.rejectAll(new Error('arena control connection closed'))
      this.emit('disconnected')
    })
    await new Promise<void>((resolve, reject) => {
      this.socket?.once('connect', resolve)
      this.socket?.once('error', reject)
    })
  }

  async request(command: string, payload: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    if (!this.socket || this.socket.destroyed) throw new Error('arena client is not connected')
    const id = ++this.sequence
    const message = JSON.stringify({ type: 'command', id, command, payload }) + '\n'
    const response = new Promise<Record<string, unknown>>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`arena command timed out: ${command}`))
      }, 10_000)
      this.pending.set(id, { resolve, reject, timeout })
    })
    this.socket.write(message)
    return response
  }

  close(): void {
    this.socket?.destroy()
    this.socket = null
    this.rejectAll(new Error('arena client closed'))
  }

  private onData(data: string): void {
    this.buffer += data
    while (true) {
      const newline = this.buffer.indexOf('\n')
      if (newline < 0) break
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (!line) continue
      try {
        const message = JSON.parse(line) as Record<string, any>
        if (message.type === 'response' && Number.isInteger(message.id)) {
          const pending = this.pending.get(message.id)
          if (!pending) continue
          clearTimeout(pending.timeout)
          this.pending.delete(message.id)
          if (message.ok) pending.resolve(message.payload ?? {})
          else pending.reject(new Error(String(message.error ?? 'arena command failed')))
        } else if (message.type === 'event') {
          this.emit('event', message as ArenaEvent)
          this.emit(String(message.event), message as ArenaEvent)
        }
      } catch (error) {
        this.emit('error', error)
      }
    }
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout)
      pending.reject(error)
    }
    this.pending.clear()
  }
}
