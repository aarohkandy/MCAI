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

  constructor(
    private readonly url: string,
    private readonly workerId: string,
    private readonly agentIds: string[],
    private readonly capabilities: string[]
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
    // The trainer is localhost-only and messages are already binary
    // MessagePack. Per-message deflate adds latency to every 50 ms decision
    // without saving network time.
    const socket = new WebSocket(this.url, { perMessageDeflate: false })
    socket.binaryType = 'arraybuffer'
    socket.on('open', () => {
      this.socket = socket
      const hello: HelloMessage = {
        schema_version: SCHEMA_VERSION,
        type: 'hello',
        sequence: 0,
        worker_id: this.workerId,
        agents: this.agentIds,
        capabilities: this.capabilities
      }
      socket.send(encode(hello))
      this.emit('ready')
    })
    socket.on('message', data => {
      try {
        const bytes = data instanceof Buffer ? data : new Uint8Array(data as ArrayBuffer)
        const message = decode(bytes) as WireMessage
        if (message.schema_version !== SCHEMA_VERSION) throw new Error('trainer uses an incompatible schema')
        if (message.type === 'action_batch') this.emit('actions', message as ActionBatch)
        else this.emit('message', message)
      } catch (error) {
        this.emit('error', error)
      }
    })
    socket.on('error', error => this.emit('error', error))
    socket.on('close', () => {
      if (this.socket === socket) this.socket = null
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
