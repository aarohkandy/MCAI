import { EventEmitter } from 'node:events'
import { decode, encode } from '@msgpack/msgpack'
import WebSocket from 'ws'
import {
  SCHEMA_VERSION,
  type ActionBatch,
  type HelloMessage,
  type StepBatch,
  type WireMessage
} from './contracts.js'

export class TrainerConnection extends EventEmitter {
  private socket: WebSocket | null = null
  private reconnectTimer: NodeJS.Timeout | null = null
  private closed = false
  private ready = false

  constructor(
    private readonly url: string,
    private readonly workerId: string,
    private readonly agentIds: string[]
  ) {
    super()
  }

  connect(): void {
    this.closed = false
    this.open()
  }

  isReady(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  sendSteps(batch: StepBatch): boolean {
    if (!this.isReady()) return false
    this.socket?.send(encode(batch))
    return true
  }

  close(): void {
    this.closed = true
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.socket?.close()
    this.socket = null
  }

  private open(): void {
    if (this.closed) return
    const socket = new WebSocket(this.url)
    socket.binaryType = 'arraybuffer'
    // Track the socket while it is still connecting so close() can tear it down; isReady()
    // already gates sends on readyState === OPEN.
    this.socket = socket
    socket.on('open', () => {
      if (this.closed) { socket.close(); return }
      const hello: HelloMessage = {
        schema_version: SCHEMA_VERSION,
        type: 'hello',
        sequence: 0,
        worker_id: this.workerId,
        agents: this.agentIds,
        capabilities: ['minecraft-1.12.2', 'structured-state', 'legal-controls-v1']
      }
      socket.send(encode(hello))
      // 'ready' is deliberately NOT emitted here. A TCP/WebSocket open only means the port
      // accepted us; the trainer may still be loading a checkpoint or unable to serve actions.
      // Emitting on open disables the scripted fallback exactly when it is still needed, leaving
      // bots idle. We emit once the trainer answers (hello_ack, or any action_batch).
    })
    socket.on('message', data => {
      try {
        const bytes = data instanceof Buffer ? data : new Uint8Array(data as ArrayBuffer)
        const message = decode(bytes) as WireMessage
        if (message.schema_version !== SCHEMA_VERSION) throw new Error('trainer uses an incompatible schema')
        // The trainer has proven it can actually serve us; only now retire the scripted fallback.
        if (!this.ready) {
          this.ready = true
          this.emit('ready')
        }
        if (message.type === 'action_batch') this.emit('actions', message as ActionBatch)
        else {
          // Surface trainer-side errors instead of silently dropping them; a control/error frame
          // is usually the only evidence that the trainer rejected our batches.
          const control = message as { command?: string; payload?: { message?: string } }
          if (control.command === 'error') {
            this.emit('error', new Error(`trainer error: ${control.payload?.message ?? 'unknown'}`))
          }
          this.emit('message', message)
        }
      } catch (error) {
        this.emit('error', error)
      }
    })
    socket.on('error', error => this.emit('error', error))
    socket.on('close', () => {
      if (this.socket === socket) this.socket = null
      this.ready = false
      this.emit('disconnected')
      this.scheduleReconnect()
    })
  }

  private scheduleReconnect(): void {
    if (this.closed || this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.open()
    }, 2_000)
  }
}
