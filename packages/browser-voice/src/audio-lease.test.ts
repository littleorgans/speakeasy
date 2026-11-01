import assert from "node:assert/strict";
import { test } from "node:test";
import { AudioWorkLease } from "./audio-lease.ts";

test("room, serial comparison group and tryout exclude each other until all work settles", () => {
  const lease = new AudioWorkLease();
  const active = lease.tryAcquire("comparison")!;
  const queued = lease.tryAcquire("comparison")!;
  assert.equal(lease.tryAcquire("room"), undefined);
  assert.equal(lease.tryAcquire("tryout"), undefined);
  active();
  active();
  assert.equal(lease.active, "comparison");
  queued();
  const room = lease.tryAcquire("room")!;
  assert.equal(lease.tryAcquire("comparison"), undefined);
  assert.equal(lease.tryAcquire("room"), undefined);
  room();
  const tryout = lease.tryAcquire("tryout")!;
  assert.equal(lease.tryAcquire("room"), undefined);
  lease.close();
  tryout();
  assert.equal(lease.active, undefined);
  assert.equal(lease.tryAcquire("comparison"), undefined);
});
