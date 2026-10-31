/**
 * Microphone constraints for every speak-easy capture page. The prompt
 * reader records bench takes with exactly these, so TM dictation, which
 * vendors this capture path, should open the mic with the same set.
 */
export const CAPTURE_AUDIO_CONSTRAINTS = Object.freeze({
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
});
