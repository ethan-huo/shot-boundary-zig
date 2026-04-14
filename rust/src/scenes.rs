use thiserror::Error;

pub type FrameIndex = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Scene {
    start: FrameIndex,
    end: FrameIndex,
}

impl Scene {
    pub fn new(start: FrameIndex, end: FrameIndex) -> Result<Self, SceneDetectionError> {
        if start > end {
            return Err(SceneDetectionError::InvalidSceneRange { start, end });
        }

        Ok(Self { start, end })
    }

    pub const fn start(self) -> FrameIndex {
        self.start
    }

    pub const fn end(self) -> FrameIndex {
        self.end
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum SceneDetectionError {
    #[error("predictions cannot be empty")]
    EmptyPredictions,

    #[error("threshold must be a finite probability in [0, 1], got {threshold}")]
    InvalidThreshold { threshold: f32 },

    #[error("prediction at frame {index} is not finite: {value}")]
    NonFinitePrediction { index: FrameIndex, value: f32 },

    #[error("scene range is invalid: start={start}, end={end}")]
    InvalidSceneRange { start: FrameIndex, end: FrameIndex },
}

pub fn predictions_to_scenes(
    predictions: &[f32],
    threshold: f32,
) -> Result<Vec<Scene>, SceneDetectionError> {
    if predictions.is_empty() {
        return Err(SceneDetectionError::EmptyPredictions);
    }

    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SceneDetectionError::InvalidThreshold { threshold });
    }

    let mut scenes = Vec::new();
    let mut previous_is_transition = false;
    let mut current_start = 0;

    for (index, prediction) in predictions.iter().copied().enumerate() {
        if !prediction.is_finite() {
            return Err(SceneDetectionError::NonFinitePrediction {
                index,
                value: prediction,
            });
        }

        let is_transition = prediction > threshold;

        if previous_is_transition && !is_transition {
            current_start = index;
        }

        if !previous_is_transition && is_transition && index != 0 {
            scenes.push(Scene::new(current_start, index)?);
        }

        previous_is_transition = is_transition;
    }

    if !previous_is_transition {
        scenes.push(Scene::new(current_start, predictions.len() - 1)?);
    }

    if scenes.is_empty() {
        scenes.push(Scene::new(0, predictions.len() - 1)?);
    }

    Ok(scenes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_scenes_between_transition_runs() {
        let scenes = predictions_to_scenes(&[0.1, 0.2, 0.8, 0.7, 0.1, 0.2], 0.5).unwrap();

        assert_eq!(
            scenes,
            vec![Scene::new(0, 2).unwrap(), Scene::new(4, 5).unwrap()]
        );
    }

    #[test]
    fn keeps_all_clear_predictions_as_one_scene() {
        let scenes = predictions_to_scenes(&[0.1, 0.2, 0.3], 0.5).unwrap();

        assert_eq!(scenes, vec![Scene::new(0, 2).unwrap()]);
    }

    #[test]
    fn mirrors_upstream_all_transition_fallback() {
        let scenes = predictions_to_scenes(&[0.9, 0.8, 0.7], 0.5).unwrap();

        assert_eq!(scenes, vec![Scene::new(0, 2).unwrap()]);
    }

    #[test]
    fn rejects_empty_predictions() {
        assert_eq!(
            predictions_to_scenes(&[], 0.5),
            Err(SceneDetectionError::EmptyPredictions)
        );
    }

    #[test]
    fn rejects_invalid_threshold() {
        assert_eq!(
            predictions_to_scenes(&[0.1], 1.1),
            Err(SceneDetectionError::InvalidThreshold { threshold: 1.1 })
        );
    }

    #[test]
    fn rejects_non_finite_predictions() {
        let error = predictions_to_scenes(&[0.1, f32::NAN], 0.5).unwrap_err();

        match error {
            SceneDetectionError::NonFinitePrediction { index, value } => {
                assert_eq!(index, 1);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
