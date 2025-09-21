from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from FitnessTrainerLSTM import FitnessTrainerLSTM, LABELS

app = FastAPI(title="Trainify API", version="1.0.0")

clf = FitnessTrainerLSTM('fitness_trainer.tflite')

class PredictRequest(BaseModel):
	# Option A: flat 36 vector (x1..x18, y1..y18)
	keypoints_36: List[float] | None = None
	# Option B: list of [id, x, y] mediapipe-style
	landmarks: List[List[float]] | None = None

class PredictResponse(BaseModel):
	label: str
	probabilities: Dict[str, float]

@app.get("/health")
def health() -> Dict[str, Any]:
	return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	if req.keypoints_36 is None and req.landmarks is None:
		raise HTTPException(status_code=400, detail="Provide keypoints_36 or landmarks")

	if req.keypoints_36 is not None:
		if len(req.keypoints_36) != 36:
			raise HTTPException(status_code=400, detail="keypoints_36 must have length 36")
		inp = req.keypoints_36
	else:
		# landmarks: list of [id, x, y]; convert to 36 vector following the integration mapping
		# Expecting 18 selected landmark ids mapping from FitnessTrainer_integration.lm_dict
		from FitnessTrainer_integration import lm_dict
		landmark_list = req.landmarks
		# Build a dict from id -> (x, y)
		id_to_xy = {int(l[0]): (float(l[1]), float(l[2])) for l in landmark_list}
		inp = []
		for index in range(0, 36):
			if index < 18:
				kp = lm_dict[index]
				inp.append(round(id_to_xy.get(kp, (0.0, 0.0))[0], 3))
			else:
				kp = lm_dict[index - 18]
				inp.append(round(id_to_xy.get(kp, (0.0, 0.0))[1], 3))

	label = clf.predict(inp)
	probs = getattr(clf, "results", dict(zip(LABELS, [0.0]*len(LABELS))))
	probs = {k: float(v) for k, v in probs.items()}
	return PredictResponse(label=label, probabilities=probs)
