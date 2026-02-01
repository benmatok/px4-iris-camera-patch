Here is the plan restructured into **4 Independent, Verifiable Phases**. Each phase has a clear "Definition of Done" so you can validate progress without needing the whole system to work at once.

### **Phase 1: The "Brain" on the Bench (Core Kernel)**

**Goal:** Validate the C++ math and physics models in isolation before touching any simulation or robot.

* **Step 1.1: Gradient Check Unit Test**
* **Action:** Write a C++ test that compares your analytical gradients (from `ghost_model.hpp`) against numerical finite differences.
* **Validation:** Assert `error < 1e-5`. If this fails, the controller will crash.


* **Step 1.2: Estimator Convergence Test**
* **Action:** Feed the `PyGhostEstimator` fake data where the "drone" suddenly becomes heavy (change mass in the feed).
* **Validation:**  Graph the probability weights. The "Heavy" model weight must cross 0.9 within 20 steps.


* **Step 1.3: Solver Latency Benchmark**
* **Action:** Run `solver.solve()` 10,000 times in a loop in Python.
* **Validation:** Average execution time must be **< 1ms**.



### **Phase 2: The "Nervous System" (Simulation Bridge)**

**Goal:** Ensure data flows correctly between Gazebo/PX4 and your Python script. Ignore control logic for now.

* **Step 2.1: Telemetry Sanity Check**
* **Action:** Subscribe to MAVSDK/ROS2 telemetry. Print `Altitude` and `IMU Z-Accel`.
* **Validation:**
* Altitude should be ~0m on ground.
* **Crucial:** Check `IMU Z`. Is it +9.8 or -9.8? This determines if you need to flip signs for your C++ kernel.




* **Step 2.2: The "Wiggle" Test (Actuation)**
* **Action:** Send a raw command: `Roll = 0.1 rad` (approx 5 degrees) for 2 seconds, then zero.
* **Validation:**  Watch Gazebo. The drone should twitch right and level out. This confirms the command pipeline works.



### **Phase 3: The "Eyes" (Perception Layer)**

**Goal:** Build the vision system using pre-recorded video or a static Gazebo camera feed.

* **Step 3.1: Red Blob Detector**
* **Action:** Implement HSV thresholding in OpenCV.
* **Validation:**  Display the camera feed. A green bounding box should accurately draw around the red object, even if you move the red object in the sim.


* **Step 3.2: Ray-Casting Math**
* **Action:** Implement the `screen_to_world` function.
* **Validation:**
* Place drone at `x=0, y=0, z=10`.
* Place red object at `x=10, y=10`.
* The function must output `Target_Pos ≈ [10, 10, 0]`.





### **Phase 4: Closed-Loop Integration (The "Gauntlet")**

**Goal:** Connect Phase 1, 2, and 3. This is the first time the drone flies autonomously.

* **Step 4.1: The "Z-Hold" Test**
* **Action:** Run the controller with `Target=[Current_X, Current_Y, 5.0]`. Ignore vision.
* **Validation:** Drone takes off and holds 5m altitude stably. If it oscillates, tune `D` gains or Estimator noise.


* **Step 4.2: The "Statue" Test (Stationary Tracking)**
* **Action:** Hover at 5m. Spawn red object at `x=10`. Enable Vision.
* **Validation:** Drone flies to `x=10` and hovers above the object.


* **Step 4.3: The "Bullfight" (Dynamic Tracking)**
* **Action:** Manually move the red object in Gazebo.
* **Validation:**  Drone chases the object. The "Future Horizon" lines (debug visualization) should point toward the moving target.
