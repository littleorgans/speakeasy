/**
 * Live words over the host's /live-words channel: while the owner holds the
 * key, a take's mic frames stream to the speech in model, and its settled and
 * still changing words come back tagged with the take's id. Messages sent
 * before the channel opens wait in order. Events reach onEvent; a closed or
 * failed channel reports an error for the take in progress.
 */
export class LiveWords {
  constructor({ onEvent = () => {} } = {}) {
    this.onEvent = onEvent;
    this.socket = null;
    this.waiting = [];
    this.takeId = null;
    this.serial = 0;
  }

  /** Starts a take on the live model with its settings, led by its pre roll; returns the take's id. */
  begin(setupId, preRoll, settings) {
    this.end();
    this.takeId = `live-${Date.now()}-${++this.serial}`;
    this.send(JSON.stringify({ type: "begin", takeId: this.takeId, setupId, ...(settings ? { settings } : {}) }));
    if (preRoll?.length) this.send(preRoll);
    return this.takeId;
  }

  push(samples) {
    if (this.takeId) this.send(samples);
  }

  /** Stops feeding the model; words already on their way still arrive. */
  end() {
    if (!this.takeId) return;
    this.send(JSON.stringify({ type: "end", takeId: this.takeId }));
    this.takeId = null;
  }

  send(message) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(message);
    else {
      this.waiting.push(message);
      this.open();
    }
  }

  open() {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/live-words`);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.addEventListener("open", () => {
      for (const message of this.waiting.splice(0)) socket.send(message);
    });
    socket.addEventListener("message", (event) => {
      try { this.onEvent(JSON.parse(event.data)); } catch { /* not ours */ }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.waiting = [];
      if (this.takeId) this.onEvent({ type: "error", takeId: this.takeId, message: "live words channel closed" });
    });
  }
}
