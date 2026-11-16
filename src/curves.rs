//! Validated legacy RED curves, including nonlinear Bezier time coordinates.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Invalid or unsupported native curve data.
#[derive(Debug, thiserror::Error)]
#[error("invalid native curve: {0}")]
pub struct CurveError(pub String);

/// Authored curve controls in their original order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Curve {
    /// Native interpolation spelling.
    pub interpolation: String,
    /// Ordered time/value control points; Bezier handles are not independent keys.
    pub points: Vec<[f64; 2]>,
}

impl Curve {
    /// Reads and validates a reflected legacy curve.
    ///
    /// # Errors
    /// Returns an error for malformed, nonfinite, unordered or unsupported controls.
    pub fn from_value(value: &Value) -> Result<Self, CurveError> {
        let text = |v: &Value| {
            v.as_str()
                .or_else(|| v.get("$value").and_then(Value::as_str))
                .map(str::to_owned)
        };
        let interpolation = text(&value["InterpolationType"])
            .ok_or_else(|| CurveError("missing interpolation".into()))?;
        let elements = value["Elements"]
            .as_array()
            .ok_or_else(|| CurveError("missing controls".into()))?;
        let points = elements
            .iter()
            .map(|p| {
                Ok([
                    p["Point"]
                        .as_f64()
                        .ok_or_else(|| CurveError("invalid control time".into()))?,
                    p["Value"]
                        .as_f64()
                        .ok_or_else(|| CurveError("invalid control value".into()))?,
                ])
            })
            .collect::<Result<Vec<_>, CurveError>>()?;
        let curve = Self {
            interpolation,
            points,
        };
        curve.validate()?;
        Ok(curve)
    }

    fn degree(&self) -> Result<usize, CurveError> {
        match self.interpolation.as_str() {
            "BezierCubic" => Ok(3),
            "BezierQuadratic" => Ok(2),
            "Linear" | "Constant" => Ok(1),
            other => Err(CurveError(format!("unsupported interpolation {other}"))),
        }
    }

    fn validate(&self) -> Result<(), CurveError> {
        let degree = self.degree()?;
        if self.points.is_empty()
            || !(self.points.len() - 1).is_multiple_of(degree)
            || self.points.iter().flatten().any(|n| !n.is_finite())
            || self.points.windows(2).any(|p| p[0][0] > p[1][0])
        {
            return Err(CurveError(
                "invalid control count, values or ordering".into(),
            ));
        }
        Ok(())
    }

    /// Evaluates y at x, clamping beyond the endpoints and solving Bezier x(t).
    ///
    /// # Errors
    /// Returns an error for invalid controls or a nonfinite input.
    pub fn evaluate(&self, x: f64) -> Result<f64, CurveError> {
        self.validate()?;
        if !x.is_finite() {
            return Err(CurveError("nonfinite input".into()));
        }
        let first = self.points[0];
        let last = self.points[self.points.len() - 1];
        if x <= first[0] {
            return Ok(first[1]);
        }
        if x >= last[0] {
            return Ok(last[1]);
        }
        let degree = self.degree()?;
        for start in (0..self.points.len() - 1).step_by(degree) {
            let controls = &self.points[start..=start + degree];
            if x > controls[degree][0] {
                continue;
            }
            if self.interpolation == "Constant" {
                return Ok(if x == controls[degree][0] {
                    controls[degree][1]
                } else {
                    controls[0][1]
                });
            }
            let bezier = |t: f64, axis: usize| {
                let mut values = [0.; 4];
                for (i, p) in controls.iter().enumerate() {
                    values[i] = p[axis];
                }
                for n in (1..controls.len()).rev() {
                    for i in 0..n {
                        values[i] = (1. - t) * values[i] + t * values[i + 1];
                    }
                }
                values[0]
            };
            let (mut low, mut high) = (0., 1.);
            for _ in 0..60 {
                let mid = (low + high) / 2.;
                if bezier(mid, 0) < x {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            return Ok(bezier((low + high) / 2., 1));
        }
        Err(CurveError("input not covered".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bezier_solves_time_and_rejects_invalid_controls() {
        let c = Curve {
            interpolation: "BezierCubic".into(),
            points: vec![[0., 0.], [0., 1.], [0., 1.], [1., 1.]],
        };
        assert!((c.evaluate(0.125).unwrap() - 0.875).abs() < 1e-10);
        assert_eq!(c.evaluate(-1.).unwrap(), 0.);
        assert!(c.evaluate(f64::NAN).is_err());
        let c = Curve {
            points: vec![[1., 0.], [0., 1.]],
            interpolation: "Linear".into(),
        };
        assert!(c.evaluate(0.5).is_err());
    }
}
