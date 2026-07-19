import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatHistory, DEFAULT_SYSTEM_PROMPT } from "./history.ts";

test("messages() pins the system prompt at the head", () => {
  const history = new ChatHistory();
  history.addUser("hello");
  const messages = history.messages();
  assert.equal(messages[0]?.role, "system");
  assert.equal(messages[0]?.content, DEFAULT_SYSTEM_PROMPT);
  assert.deepEqual(messages[1], { role: "user", content: "hello" });
});

test("history caps the turn window and drops the oldest", () => {
  const history = new ChatHistory("sys", 2);
  history.addUser("one");
  history.addAssistant("two");
  history.addUser("three");
  assert.equal(history.size, 2);
  const turns = history.messages().slice(1);
  assert.deepEqual(turns, [
    { role: "assistant", content: "two" },
    { role: "user", content: "three" },
  ]);
});

test("a custom system prompt is used", () => {
  const history = new ChatHistory("be terse", 4);
  assert.equal(history.messages()[0]?.content, "be terse");
});
