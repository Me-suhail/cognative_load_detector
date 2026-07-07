1. Recommended Main Research States

->Fatigued / Drowsy
- Supported by: eye closure, PERCLOS, blink rate, micro-sleep, drowsiness detection papers.
- Good app evidence: EAR, PERCLOS, blink rate, eye closure duration, micro-sleep.
- Research use: Strong state to keep.

->Distracted / Inattentive
- Supported by: gaze direction, off-screen time, head pose, attention and engagement papers.
- Good app evidence: off-screen percentage, head turned away, gaze instability.
- Research use: Strong state to keep.

->High Cognitive Load / Overloaded
- Supported by: cognitive load and mental workload papers using eye metrics, gaze, pupil/blink behavior, and head/behavioral signals.
- Good app evidence: unstable gaze, abnormal blink rate, low eye openness, high brow/strain signals, reduced focus score.
- Research use: Keep, but rename "overloaded" to "high cognitive load" in the research paper.

->Focused / Engaged
- Supported by: engagement and attention monitoring papers.
- Good app evidence: stable gaze, normal blink rate, steady head pose, good eye openness, high focus score.
- Research use: Keep, but "engaged" is usually more research-friendly than only "focused".

2. Use Carefully

->Light Load
- Supported by: cognitive load theory, but weak if detected only from webcam signals.
- Good app evidence: good focus score but not intense/high-load behavior.
- Research use: Use as "low cognitive load" only if the task difficulty or questionnaire supports it. Otherwise merge with focused/engaged.

3. Weak as a Main Research State

->Confused
- Supported by: weak direct support from facial/eye features alone.
- Problem: Confusion is a mental state and is hard to prove only from blink, gaze, brow, or emotion.
- Research use: Avoid as a main class unless users self-report confusion or task performance confirms it.

4. Best Final State Set

->Engaged / Focused
- Purpose: Normal productive state.

->Distracted / Inattentive
- Purpose: Attention is away from screen/task.

->High Cognitive Load
- Purpose: User may be mentally overloaded or under heavy task demand.

->Fatigued / Drowsy
- Purpose: User shows fatigue-related eye and attention signals.

5. Burnout Risk Should Be Separate

->Low Burnout Risk
- Based on: validated burnout questionnaire score.

->Moderate Burnout Risk
- Based on: validated burnout questionnaire score plus repeated fatigue/high-load indicators.

->High Burnout Risk
- Based on: validated burnout questionnaire score plus repeated fatigue/high-load indicators.

6. Papers To Support These States

->PERCLOS-based technologies for detecting drowsiness
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/
- Purpose: Strong support for fatigue/drowsiness using eye closure and PERCLOS.

->A Review of the Use of Gaze and Pupil Metrics to Assess Mental Workload
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10975796/
- Purpose: Supports using gaze and pupil metrics for cognitive load / mental workload.

->Eye Tracking and Head Movement Detection: A State-of-Art Survey
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC4839304/
- Purpose: Supports gaze tracking and head movement features used in attention/distracted states.

->Estimation of behavioral user state based on eye gaze and head pose in an e-learning environment
- Link: https://doi.org/10.1007/s11042-008-0240-1
- Purpose: Supports using eye gaze and head pose for attention/engagement state estimation in e-learning.

7. Simple Research Position

The app should not claim to directly detect burnout from webcam signals.

The stronger claim is:
The system detects short-term cognitive and behavioral states such as engagement, distraction, high cognitive load, and fatigue. These are then combined with a validated burnout questionnaire to estimate burnout risk.

