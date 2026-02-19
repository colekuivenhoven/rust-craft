// Simple fire-and-forget audio manager.
// Wraps rodio's OutputStream so callers only need to call `play(path)`.
// The _stream field MUST be kept alive for the duration of the program;
// dropping it silences the audio device.

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use std::fs::File;
use std::io::BufReader;

pub struct AudioManager {
    _stream: OutputStream,
    handle:  OutputStreamHandle,
}

impl AudioManager {
    /// Attempt to open the default audio output device.
    /// Returns `None` (with a log warning) if no device is available.
    pub fn new() -> Option<Self> {
        match OutputStream::try_default() {
            Ok((_stream, handle)) => Some(Self { _stream, handle }),
            Err(e) => {
                log::warn!("Audio init failed, sounds disabled: {}", e);
                None
            }
        }
    }

    /// Create a looping sink for `path` at initial volume 0.0.
    /// The caller owns the `Sink` and controls volume / playback.
    pub fn create_looping_sink(&self, path: &str) -> Option<Sink> {
        let file = match File::open(path) {
            Ok(f)  => f,
            Err(e) => { log::warn!("Cannot open '{}': {}", path, e); return None; }
        };
        let source = match Decoder::new(BufReader::new(file)) {
            Ok(s)  => s,
            Err(e) => { log::warn!("Cannot decode '{}': {}", path, e); return None; }
        };
        let sink = match Sink::try_new(&self.handle) {
            Ok(s)  => s,
            Err(e) => { log::warn!("Cannot create sink: {}", e); return None; }
        };
        sink.set_volume(0.0);
        sink.append(source.repeat_infinite());
        Some(sink)
    }

    /// Play a music file at `path` that loops indefinitely.
    /// The sink is detached so it runs for the lifetime of the audio device.
    pub fn play_looping(&self, path: &str) {
        let file = match File::open(path) {
            Ok(f)  => f,
            Err(e) => { log::warn!("Cannot open music '{}': {}", path, e); return; }
        };
        let source = match Decoder::new(BufReader::new(file)) {
            Ok(s)  => s,
            Err(e) => { log::warn!("Cannot decode music '{}': {}", path, e); return; }
        };
        let sink = match Sink::try_new(&self.handle) {
            Ok(s)  => s,
            Err(e) => { log::warn!("Cannot create music sink: {}", e); return; }
        };
        sink.append(source.repeat_infinite());
        sink.detach();
    }

    /// Play a sound file at `path` (relative to the working directory).
    /// Plays fire-and-forget; errors are logged and silently ignored.
    pub fn play(&self, path: &str) {
        let file = match File::open(path) {
            Ok(f)  => f,
            Err(e) => { log::warn!("Cannot open sound '{}': {}", path, e); return; }
        };
        let source = match Decoder::new(BufReader::new(file)) {
            Ok(s)  => s,
            Err(e) => { log::warn!("Cannot decode sound '{}': {}", path, e); return; }
        };
        if let Err(e) = self.handle.play_raw(source.convert_samples()) {
            log::warn!("Cannot play sound '{}': {}", path, e);
        }
    }
}
