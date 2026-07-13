(function MCAICoordinateContract(global) {
  'use strict'

  const radians = value => Number(value || 0) * Math.PI / 180
  const degrees = value => Number(value || 0) * 180 / Math.PI
  const normalize = value => {
    let angle = Number(value || 0)
    while (angle > Math.PI) angle -= Math.PI * 2
    while (angle < -Math.PI) angle += Math.PI * 2
    return angle
  }

  global.MCAICoordinates = Object.freeze({
    canonicalYaw(minecraftYawDegrees) { return normalize(Math.PI - radians(minecraftYawDegrees)) },
    canonicalPitch(minecraftPitchDegrees) { return -radians(minecraftPitchDegrees) },
    minecraftYawDelta(canonicalDeltaRadians) { return -degrees(canonicalDeltaRadians) },
    minecraftPitchDelta(canonicalDeltaRadians) { return -degrees(canonicalDeltaRadians) }
  })
})(typeof globalThis !== 'undefined' ? globalThis : window)
