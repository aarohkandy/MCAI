import fs from 'node:fs/promises'

const [, , destination, port = '9337'] = process.argv
if (!destination) throw new Error('usage: node capture-spectator.mjs <destination.png> [debug-port]')

const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
const page = pages.find(value => value.type === 'page')
if (!page?.webSocketDebuggerUrl) throw new Error('spectator page is not available')

const screenshot = await new Promise((resolve, reject) => {
  const socket = new WebSocket(page.webSocketDebuggerUrl)
  socket.onerror = reject
  socket.onopen = () => socket.send(JSON.stringify({
    id: 1,
    method: 'Page.captureScreenshot',
    params: { format: 'png', fromSurface: true }
  }))
  socket.onmessage = event => {
    const message = JSON.parse(event.data)
    if (message.id !== 1) return
    socket.close()
    if (message.error) reject(new Error(message.error.message))
    else resolve(Buffer.from(message.result.data, 'base64'))
  }
})

await fs.writeFile(destination, screenshot)
