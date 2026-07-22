const [, , command, port = '9337'] = process.argv
if (!command) throw new Error('usage: node send-spectator-command.mjs <command> [debug-port]')

const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
const page = pages.find(value => value.type === 'page')
if (!page?.webSocketDebuggerUrl) throw new Error('spectator page is not available')

const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.onopen = resolve
  socket.onerror = reject
})

let messageId = 0
const pending = new Map()
socket.onmessage = event => {
  const message = JSON.parse(event.data)
  const handler = pending.get(message.id)
  if (!handler) return
  pending.delete(message.id)
  if (message.error) handler.reject(new Error(message.error.message))
  else handler.resolve(message.result)
}
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++messageId
  pending.set(id, { resolve, reject })
  socket.send(JSON.stringify({ id, method, params }))
})
const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))

await send('Page.bringToFront')
const layout = await send('Page.getLayoutMetrics')
const width = layout.cssVisualViewport?.clientWidth ?? 800
const height = layout.cssVisualViewport?.clientHeight ?? 800
await send('Input.dispatchMouseEvent', {
  type: 'mousePressed', x: width / 2, y: height / 2, button: 'left', clickCount: 1
})
await send('Input.dispatchMouseEvent', {
  type: 'mouseReleased', x: width / 2, y: height / 2, button: 'left', clickCount: 1
})
await send('Input.dispatchKeyEvent', {
  type: 'rawKeyDown', key: 't', code: 'KeyT', windowsVirtualKeyCode: 84,
  nativeVirtualKeyCode: 84
})
await send('Input.dispatchKeyEvent', {
  type: 'keyUp', key: 't', code: 'KeyT', windowsVirtualKeyCode: 84
})
await wait(500)
for (const character of command) {
  const letter = /[a-z]/i.test(character)
  const code = letter ? `Key${character.toUpperCase()}` : (character === '/' ? 'Slash' : '')
  const virtualKeyCode = letter ? character.toUpperCase().charCodeAt(0) : (character === '/' ? 191 : 0)
  await send('Input.dispatchKeyEvent', {
    type: 'keyDown', key: character, code, text: character, unmodifiedText: character,
    windowsVirtualKeyCode: virtualKeyCode, nativeVirtualKeyCode: virtualKeyCode
  })
  await send('Input.dispatchKeyEvent', {
    type: 'keyUp', key: character, code, windowsVirtualKeyCode: virtualKeyCode,
    nativeVirtualKeyCode: virtualKeyCode
  })
  await wait(25)
}
await wait(300)
await send('Input.dispatchKeyEvent', {
  type: 'rawKeyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13,
  nativeVirtualKeyCode: 13
})
await send('Input.dispatchKeyEvent', {
  type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13
})
await wait(1500)
socket.close()

console.log(`sent ${command} through Eaglercraft`)
