// WebSocket to the laptop with reconnect (backoff 250 ms doubling to 2 s). `onmessage` gets the raw data.
export function openSocket(path, { binary = false, onopen, onmessage, onclose } = {}) {
  let ws = null;
  let delay = 250;
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${path}`;

  function connect() {
    const sock = new WebSocket(url);
    if (binary) sock.binaryType = 'arraybuffer';
    sock.onopen = () => {
      ws = sock;
      delay = 250;
      if (onopen) onopen();
    };
    sock.onmessage = (e) => onmessage(e.data);
    sock.onclose = () => {
      if (ws === sock) ws = null;
      if (onclose) onclose();
      setTimeout(connect, delay);
      delay = Math.min(2000, delay * 2);
    };
  }
  connect();

  return {
    send(data) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(typeof data === 'string' || data instanceof ArrayBuffer ? data : JSON.stringify(data));
        return true;
      }
      return false;
    },
    get connected() {
      return ws !== null && ws.readyState === WebSocket.OPEN;
    },
    // Start over: the connection looks up but nothing comes (the other side waits for something this page never sent).
    reconnect() {
      if (ws) ws.close();
    },
  };
}
