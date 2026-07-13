import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { NOOP_ACTION, type ActionV1, validateAction } from './contracts.js'
import { clamp, normalizeAngle } from './math.js'
import type { ControlTelemetry } from './observation.js'

export type ControlAuditEvent = {
  tick: number
  operation: string
  detail?: string
}

export class LegalControlAdapter {
  private tick = 0
  private activeHand: ControlTelemetry['activeHand'] = 'none'
  private useStartedTick = 0
  private lastAttackTick = -1000
  private miningProgress = 0
  private miningStartedTick = 0
  private miningExpectedTicks = 1
  private digging = false
  private lastAction: ActionV1 = { ...NOOP_ACTION }
  private audit: ControlAuditEvent[] = []

  constructor(private readonly bot: Bot) {}

  telemetry(tick = this.tick): ControlTelemetry {
    if (this.digging) {
      this.miningProgress = clamp((tick - this.miningStartedTick) / this.miningExpectedTicks, 0.01, 0.99)
    }
    return {
      lastAttackTick: this.lastAttackTick,
      activeHand: this.activeHand,
      useStartedTick: this.useStartedTick,
      miningProgress: this.miningProgress
    }
  }

  getAuditEvents(): readonly ControlAuditEvent[] {
    return this.audit
  }

  async apply(input: ActionV1, tick: number): Promise<void> {
    const action = validateAction(input)
    this.tick = tick
    this.applyMovement(action)
    await this.applyCamera(action)
    if (action.hotbar >= 0 && action.hotbar !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(action.hotbar)
      this.record('hotbar', String(action.hotbar))
    }
    if (action.release_use) this.releaseUse()
    if (action.swap_offhand && !this.lastAction.swap_offhand) this.swapOffhand()
    await this.applyPrimary(action)
    this.lastAction = action
  }

  emergencyStop(): void {
    this.bot.clearControlStates()
    this.releaseUse()
    if (this.digging) this.bot.stopDigging()
    this.digging = false
    this.miningProgress = 0
    this.lastAction = { ...NOOP_ACTION }
    this.record('emergency_stop')
  }

  private applyMovement(action: ActionV1): void {
    this.bot.setControlState('forward', action.forward === 1)
    this.bot.setControlState('back', action.forward === -1)
    this.bot.setControlState('left', action.strafe === -1)
    this.bot.setControlState('right', action.strafe === 1)
    this.bot.setControlState('jump', action.jump)
    this.bot.setControlState('sprint', action.sprint)
    this.bot.setControlState('sneak', action.sneak)
  }

  private async applyCamera(action: ActionV1): Promise<void> {
    if (action.yaw_delta === 0 && action.pitch_delta === 0) return
    const yaw = normalizeAngle((this.bot.entity?.yaw ?? 0) + action.yaw_delta)
    const pitch = clamp((this.bot.entity?.pitch ?? 0) + action.pitch_delta, -Math.PI / 2, Math.PI / 2)
    await this.bot.look(yaw, pitch, true)
    this.record('look')
  }

  private async applyPrimary(action: ActionV1): Promise<void> {
    if (action.primary !== 'attack') {
      if (this.digging) {
        this.bot.stopDigging()
        this.digging = false
        this.record('stop_digging')
      }
      this.miningProgress = 0
    }
    if (action.primary === 'attack') await this.attackOrMine()
    else if (action.primary === 'use_main' && this.lastAction.primary !== 'use_main') await this.useHand(false)
    else if (action.primary === 'use_offhand' && this.lastAction.primary !== 'use_offhand') await this.useHand(true)
  }

  private async attackOrMine(): Promise<void> {
    const entity = this.bot.entityAtCursor?.(3.0)
    if (entity) {
      this.bot.attack(entity)
      this.lastAttackTick = this.tick
      this.record('attack_entity', String(entity.id))
      return
    }
    const block = this.crosshairBlock()?.block
    if (!block || this.digging || !block.diggable) return
    this.digging = true
    this.miningProgress = 0.01
    this.miningStartedTick = this.tick
    const expectedMilliseconds = Number(this.bot.digTime?.(block) ?? 50)
    this.miningExpectedTicks = Math.max(1, Math.ceil(expectedMilliseconds / 50))
    this.record('start_digging', `${block.position.x},${block.position.y},${block.position.z}`)
    void this.bot.dig(block, 'ignore').then(() => {
      this.digging = false
      this.miningProgress = 1
    }).catch(() => {
      this.digging = false
      this.miningProgress = 0
    })
  }

  private async useHand(offhand: boolean): Promise<void> {
    const item = offhand ? this.bot.inventory.slots[45] : this.bot.heldItem
    const name = String(item?.name ?? '')
    const target = this.crosshairBlock()
    if (!offhand && target && name === 'obsidian') {
      await (this.bot as any)._placeBlockWithOptions(target.block, target.face, {
        forceLook: 'ignore', swingArm: 'right', delta: target.cursor
      })
      this.record('place_from_crosshair', name)
      return
    }
    if (!offhand && target && name.includes('crystal')) {
      await (this.bot as any)._genericPlace(target.block, target.face, {
        forceLook: 'ignore', swingArm: 'right', delta: target.cursor
      })
      this.record('activate_block_from_crosshair', name)
      return
    }
    this.bot.activateItem(offhand)
    this.activeHand = offhand ? 'off' : 'main'
    this.useStartedTick = this.tick
    this.record(offhand ? 'use_offhand' : 'use_main', name)
  }

  private crosshairBlock(): { block: any; face: Vec3; cursor: Vec3 } | null {
    const entity = this.bot.entity
    if (!entity) return null
    const yaw = entity.yaw ?? 0
    const pitch = entity.pitch ?? 0
    const cosPitch = Math.cos(pitch)
    const direction = new Vec3(-Math.sin(yaw) * cosPitch, Math.sin(pitch), -Math.cos(yaw) * cosPitch)
    const eye = entity.position.offset(0, (entity as any).eyeHeight ?? 1.62, 0)
    const hit = (this.bot.world as any).raycast(eye, direction, 5.0)
    if (!hit?.position) return null
    const block = this.bot.blockAt(hit.position, false)
    if (!block) return null
    const faces = [
      new Vec3(0, -1, 0), new Vec3(0, 1, 0), new Vec3(0, 0, -1),
      new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(1, 0, 0)
    ]
    const face = faces[Number(hit.face)] ?? new Vec3(0, 1, 0)
    const intersect = hit.intersect ?? hit.position.offset(0.5, 0.5, 0.5)
    const cursor = new Vec3(
      clamp(intersect.x - hit.position.x, 0, 1),
      clamp(intersect.y - hit.position.y, 0, 1),
      clamp(intersect.z - hit.position.z, 0, 1)
    )
    return { block, face, cursor }
  }

  private releaseUse(): void {
    this.bot.deactivateItem()
    this.activeHand = 'none'
    this.record('release_use')
  }

  /**
   * Minecraft 1.12 has no Mineflayer high-level swap-key method. This is the
   * one allowlisted raw write: status 6 is the vanilla SWAP_ITEM_WITH_OFFHAND
   * digging action generated by the F key. No coordinates or motion are sent.
   */
  private swapOffhand(): void {
    ;(this.bot as any)._client.write('block_dig', {
      status: 6,
      location: new Vec3(0, 0, 0),
      face: 0
    })
    this.record('swap_offhand_key')
  }

  private record(operation: string, detail?: string): void {
    this.audit.push({ tick: this.tick, operation, detail })
    if (this.audit.length > 2048) this.audit.splice(0, 512)
  }
}
