# App Goal
- to answer the question: "how far can I travel away from my mid-range of motion, at each angle of available motion?"
# Code Pipeline
- *as of Jan 2024*
- take in video of CAR
	- *CAR start/stop must manually trimmed*
	-  (in most formats, see [OpenCV docs for more info](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a85b55cf6a4a50451367ba96b65218ba1))
- set-up PoseEstimator w/ options, via MediaPipe
- use OpenCV to go frame-by-frame and feed the frame to the PoseEstimator
	- build a data structure w/ real-coordinates (x,y,z, plus visibility score, and presence score), normalized coordinates and timestamp
- serialize this to JSON for standardization and easier storage
- then run this json through a series of analytcal processes
	- **user must supply target and moving joint ID**
	- aka `full_flow` function
	- unpack json (into python dict)
		- this dict will get passed in AND returned in each subsequent function, to slowly build a rich data
	- process landmarks: "fix" the target joint in a single spot, and normalize the moving joint around this fixed target joint appropriately
		- output normalized mj and tj points (lose the rest), as well as avg radius (to use for later calculation), mj/tj joint ids
	- smooth the points (using sliding window technique) to improve the variation of outliers
	- calc/add the centroid of the smoothed moving joint path, aka, the "middle" point of all the moving points
		- this is critical to help us later determine the displacement (from this middle) that each point represents
	- sort the points by angle: this will clean up any incoherent ordering of points (imagine: a little curlycue shape in the CAR)
	- partition the points into zones (corresponding to named zones in FRS)
	- partition the points by angle (of rotation about the centroid), which *also* calculates displacement
		- possible refactor here to further clarify
		- **partition by displacement**: works through each angular bucket (that has values) and calculates the displacement from the centroid ("how far can I travel away from the target joint"); this processing returns a set of data for each angular bucket (FORMAT of output elements: `[min_norm, Max_norm, n_of_points, (x,y,z)]`, the coord is of the max)
	- outputs a rich dictionary with results from each of these steps
# Dev goals
- build in auto-detection
	- determine start/end of CAR without need to manually trim
	- determine moving joint (based on lack of movement of other joints)
		- use the result as target/moving joint ids in subsequent processing
	- determine quality of input
		- reject if all joints aren't in frame, etc
		- would need to probe this a bit
- wrap functionality up into python library toward open_source usage of underlying processing
	- input vid file, output results dict
	- optional inputs (before auto-detection is built): moving/target joint ID
- simplify the output into format understandable by laypeople
	- step one: use time-series of tests to orient results
		- "This CARs input shows a 15% increase in 'reachable area' since your last test!"
		- "Your ability to reach (specific movement) has improved by 7%!"
	- step two: correlate CARs results across population
		- "Your CAR score puts you in the 84th percentile, given your demographics"
		- "It appears you are limited in (zone or direction of movement)"
		- "Your access to (zone) is in the 21st percentile; would you like to incoporate training this area into your program?"
- convert Mediapipe data into blender for 3d viz?
	- [thread about doing this](https://blenderartists.org/t/mediapipe-for-blender-python-programming/1312620?page=2)