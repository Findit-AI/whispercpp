#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

mod context;
mod error;
mod lang;
mod params;
mod state;
mod sys;

pub use context::{
  AlignmentHeadsPreset, Context, ContextParams, DEFAULT_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE,
  MIN_DTW_MEM_SIZE, SUPPORTED_DTW_N_TEXT_CTX, required_dtw_mem_size_for, system_info,
};
pub use error::{WhisperError, WhisperResult};
pub use lang::Lang;
pub use params::{
  MAX_BEAM_SIZE, MAX_INITIAL_TS_S, MAX_N_THREADS, MAX_TEMPERATURE, MIN_TEMPERATURE_INC, Params,
  SamplingStrategy,
};
pub use state::{Segment, State, Token};
