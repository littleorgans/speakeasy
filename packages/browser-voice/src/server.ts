import { loadEnvFile } from "node:process";
import { startBrowserVoiceServer } from "./host.ts";
import { readBrowserVoiceOptions } from "./server-options.ts";

loadLocalEnv();
const options = readBrowserVoiceOptions(process.argv.slice(2), process.env);
const host = await startBrowserVoiceServer(options);
console.log(`speak-easy browser voice: ${host.url}`);
console.log("Open the URL, allow microphone access, then start a room.");

/** Load the gitignored local environment without printing or serializing it. */
function loadLocalEnv(): void {
  try {
    loadEnvFile(".env");
  } catch (error) {
    if (!isMissingFile(error)) throw error;
  }
}

function isMissingFile(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT"
  );
}
