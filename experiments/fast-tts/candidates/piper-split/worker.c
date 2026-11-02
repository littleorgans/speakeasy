// Resident libpiper worker for the piper-split candidate: the native worker
// protocol of ../driver.ts (NativeWorker). Each request part is one sentence of
// phoneme characters; the voice config sets phoneme_type "text", so libpiper maps
// the characters to ids itself and never runs eSpeak. With the encoder and
// decoder split next to the model path, every piper_synthesize_next call returns
// one decoder chunk (first 32 latent frames, 12 frames of context each side).
//
// usage: piper-split-worker <model.onnx> <config.json>

#include <piper.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { READY = 1, AUDIO = 2, END = 3, ERROR = 4 };

static int frames_fd;

static void write_all(const void *data, size_t size) {
  const char *bytes = data;
  while (size > 0) {
    ssize_t written = write(frames_fd, bytes, size);
    if (written <= 0) exit(3);
    bytes += written;
    size -= (size_t)written;
  }
}

static void frame(int32_t kind, int32_t count, const void *payload, size_t bytes) {
  int32_t header[2] = {kind, count};
  write_all(header, sizeof header);
  if (bytes > 0) write_all(payload, bytes);
}

static void info(int32_t kind, const char *json) {
  frame(kind, (int32_t)strlen(json), json, strlen(json));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <model.onnx> <config.json>\n", argv[0]);
    return 2;
  }
  // Frames own the original stdout; anything else printed goes to stderr.
  frames_fd = dup(STDOUT_FILENO);
  dup2(STDERR_FILENO, STDOUT_FILENO);

  piper_synthesizer *synth = piper_create(argv[1], argv[2], NULL);
  if (synth == NULL) {
    info(ERROR, "{\"error\":\"piper_create failed\"}");
    return 1;
  }
  char ready[128];
  snprintf(ready, sizeof ready, "{\"piper\":\"%s\"}", piper_version());
  info(READY, ready);

  char *line = NULL;
  size_t capacity = 0;
  ssize_t length;
  while ((length = getline(&line, &capacity, stdin)) > 0) {
    if (line[length - 1] == '\n') line[--length] = '\0';
    if (strcmp(line, "Q") == 0) break;
    if (strncmp(line, "S\t", 2) != 0) {
      info(ERROR, "{\"error\":\"bad request\"}");
      continue;
    }
    int chunks = 0, failed = 0;
    for (char *part = line + 2, *next; part != NULL && !failed; part = next) {
      next = strchr(part, '\x1f');
      if (next != NULL) *next++ = '\0';
      if (piper_synthesize_start(synth, part, NULL) != PIPER_OK) {
        failed = 1;
        break;
      }
      piper_audio_chunk chunk;
      int rc;
      // Cancellation checkpoint: between these calls, one decoder chunk apart.
      // A split voice returns its last chunk's samples with PIPER_DONE.
      do {
        rc = piper_synthesize_next(synth, &chunk);
        if (rc < 0) break;
        if (chunk.num_samples == 0) continue;
        frame(AUDIO, (int32_t)chunk.num_samples, chunk.samples, chunk.num_samples * sizeof(float));
        chunks += 1;
      } while (rc == PIPER_OK);
      if (rc != PIPER_DONE) failed = 1;
    }
    if (failed) {
      info(ERROR, "{\"error\":\"synthesis failed\"}");
      continue;
    }
    char end[64];
    snprintf(end, sizeof end, "{\"chunks\":%d}", chunks);
    info(END, end);
  }
  free(line);
  piper_free(synth);
  return 0;
}
