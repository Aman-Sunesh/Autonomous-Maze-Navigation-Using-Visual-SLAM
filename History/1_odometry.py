from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np


def compose_pose(R_w_c, t_w_c, R_c1_c2, t_c1_c2):
    R_w_c2 = R_w_c @ R_c1_c2
    t_w_c2 = R_w_c @ t_c1_c2 + t_w_c
    return R_w_c2, t_w_c2


class MonoSLAM:
    """
    Monocular VO + simple mapping.

    rot_only=True:
      - KEEP EXACTLY the same logic you had:
        rotation from homography (RANSAC), translation forced to zero.

    rot_only=False:
      - TURN OFF epipolar (no EssentialMat / recoverPose).
      - Use commanded speed dead-reckoning (v_cmd, w_cmd) integrated by dt.
    """

    def __init__(self, K: np.ndarray):
        self.K = np.array(K, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)

        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        # world <- camera
        self.R_w_c = np.eye(3, dtype=np.float64)
        self.t_w_c = np.zeros((3, 1), dtype=np.float64)

        self.traj = []       # (x,z)
        self.map_points = [] # Nx3

        # Debug
        self.last_mode = "init"
        self.last_ok = False
        self.last_matches = 0
        self.last_inliers_E = 0
        self.last_inliers_H = 0
        self.last_med_flow = 0.0
        self.last_rE = 0.0
        self.last_rH = 0.0

        # thresholds (still used for match quality / homography)
        self.min_matches = 80
        self.H_ransac_thresh_px = 2.0

        self.max_map_points = 20000
        self.max_traj_len = 20000

    # ---------- helpers ----------
    def _detect(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)
        return kp, des

    def _match(self, des1, des2):
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []
        knn = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def _append_traj(self):
        x = float(self.t_w_c[0, 0])
        z = float(self.t_w_c[2, 0])
        self.traj.append((x, z))
        if len(self.traj) > self.max_traj_len:
            self.traj = self.traj[-self.max_traj_len:]

    def _compose_pose(self, R_c1_c2, t_c1_c2):
        # correct composition:
        R_old = self.R_w_c
        t_old = self.t_w_c
        self.R_w_c = R_old @ R_c1_c2
        self.t_w_c = R_old @ t_c1_c2 + t_old

    def _rotation_from_homography(self, H):
        # H ≈ K R K^{-1}  => R ≈ K^{-1} H K
        R_approx = self.Kinv @ H @ self.K
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    # ---------- main ----------
    def process_frame(self, bgr, rot_only: bool = False, dt: float = 0.0, v_cmd: float = 0.0, w_cmd: float = 0.0):
        """
        rot_only=True  => SAME as before (homography rotation, t=0).
        rot_only=False => dead-reckoning from commanded speed (no epipolar).
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._detect(gray)

        # init
        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self._append_traj()
            self.last_mode = "init"
            self.last_ok = True
            self.last_matches = 0
            self.last_inliers_E = 0
            self.last_inliers_H = 0
            self.last_med_flow = 0.0
            self.last_rE = 0.0
            self.last_rH = 0.0
            return True

        matches = self._match(self.prev_des, des)
        self.last_matches = len(matches)

        # reset debug each frame
        self.last_inliers_E = 0
        self.last_rE = 0.0
        self.last_med_flow = 0.0
        self.last_inliers_H = 0
        self.last_rH = 0.0

        # If matching is bad, freeze
        if self.last_matches < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self._append_traj()
            self.last_mode = "bad"
            self.last_ok = False
            return False

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_cur  = np.float32([kp[m.trainIdx].pt for m in matches])

        # -------------------------------------------------------------------
        # ROTATION-ONLY MODE (KEEP EXACTLY YOUR OLD LOGIC)
        # -------------------------------------------------------------------
        if rot_only:
            H, maskH = cv2.findHomography(
                pts_prev, pts_cur, method=cv2.RANSAC,
                ransacReprojThreshold=float(self.H_ransac_thresh_px)
            )
            if H is None or maskH is None:
                # can't estimate; freeze
                self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
                self._append_traj()
                self.last_mode = "rot_only_fail"
                self.last_ok = False
                return False

            inliersH = int(maskH.ravel().sum())
            self.last_inliers_H = inliersH
            self.last_rH = inliersH / max(1, self.last_matches)

            # rotation from H; translation forced 0
            R_1_2 = self._rotation_from_homography(H)

            # cam_prev <- cam_cur
            R_c1_c2 = R_1_2.T
            t_c1_c2 = np.zeros((3, 1), dtype=np.float64)

            self._compose_pose(R_c1_c2, t_c1_c2)
            self._append_traj()

            # Update reference
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self.last_mode = "rot_only"
            self.last_ok = True
            return True

        # -------------------------------------------------------------------
        # GENERAL MOTION MODE (NO EPIPOLAR): dead-reckoning from v_cmd, w_cmd
        # -------------------------------------------------------------------
        dt = float(max(0.0, min(float(dt), 0.1)))

        # Yaw rotation about +Y (world up) using commanded angular velocity
        dtheta = float(w_cmd) * dt
        c = float(np.cos(dtheta))
        s = float(np.sin(dtheta))

        # R_1_2: cam_prev -> cam_cur (pure yaw)
        R_1_2 = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ], dtype=np.float64)

        # Translation: forward along +Z in OpenCV camera frame
        t_1_2 = np.array([[0.0], [0.0], [float(v_cmd) * dt]], dtype=np.float64)

        # Convert to cam_prev <- cam_cur convention (inverse)
        R_c1_c2 = R_1_2.T
        t_c1_c2 = -R_1_2.T @ t_1_2

        self._compose_pose(R_c1_c2, t_c1_c2)
        self._append_traj()

        # Update reference (keeps matching healthy / debug meaningful)
        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        self.last_mode = "dead_reckon"
        self.last_ok = True
        return True

    def render_topdown_map(self, W, H):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if not self.traj:
            return canvas

        xs = [p[0] for p in self.traj]
        zs = [p[1] for p in self.traj]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)

        pad = 0.5
        x_min -= pad; x_max += pad
        z_min -= pad; z_max += pad

        dx = max(1e-6, x_max - x_min)
        dz = max(1e-6, z_max - z_min)
        scale = 0.9 * min(W / dx, H / dz)

        def to_px(x, z):
            u = int((x - x_min) * scale + 0.05 * W)
            v = int((z - z_min) * scale + 0.05 * H)
            v = H - v
            return u, v

        for i in range(1, len(self.traj)):
            u1, v1 = to_px(*self.traj[i - 1])
            u2, v2 = to_px(*self.traj[i])
            cv2.line(canvas, (u1, v1), (u2, v2), (255, 255, 255), 2)

        u, v = to_px(*self.traj[-1])
        cv2.circle(canvas, (u, v), 5, (255, 0, 0), -1)

        txt = f"mode={self.last_mode} ok={self.last_ok} m={self.last_matches} H_in={self.last_inliers_H} rH={self.last_rH:.2f}"
        cv2.putText(canvas, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas


class KeyboardSLAMPlayer(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None

        self.K = np.array([
            [92.0,  0.0, 160.0],
            [ 0.0, 92.0, 120.0],
            [ 0.0,  0.0,   1.0]
        ], dtype=np.float64)

        self.slam = None
        self._printed_first_see = False
        self._last_phase = None
        super().__init__()

        # Commanded speeds (match sim nominal; adjust if your sim differs)
        self.V_TRANS = 3.0  # m/s
        self.W_ROT   = 6.0  # rad/s

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def pre_navigation(self):
        super().pre_navigation()
        try:
            temp_K = self.get_camera_intrinsic_matrix()
        except Exception:
            temp_K = None

        if temp_K is not None:
            self.K = np.array(temp_K, dtype=np.float64)

        self.slam = MonoSLAM(self.K)

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
        H, W = fpv.shape[:2]
        map_w = W // 2
        map_h = H

        if self.screen is None:
            self.screen = pygame.display.set_mode((W + map_w, H))

        # Determine rot_only command exactly like your original logic
        has_left  = bool(self.last_act & Action.LEFT)
        has_right = bool(self.last_act & Action.RIGHT)
        has_fwd   = bool(self.last_act & Action.FORWARD)
        has_back  = bool(self.last_act & Action.BACKWARD)
        rot_cmd = (has_left or has_right) and (not has_fwd) and (not has_back)

        # dt from simulator fps
        fps = float(self._state[4]) if (self._state is not None and len(self._state) >= 5) else 0.0
        dt = (1.0 / fps) if fps > 1e-3 else 0.0

        # commanded speeds
        v_cmd = 0.0
        w_cmd = 0.0
        if has_fwd:
            v_cmd += self.V_TRANS
        if has_back:
            v_cmd -= self.V_TRANS
        if has_left:
            w_cmd += self.W_ROT
        if has_right:
            w_cmd -= self.W_ROT

        # Run SLAM: rot_only keeps your homography logic, else dead-reckon
        self.slam.process_frame(fpv, rot_only=rot_cmd, dt=dt, v_cmd=v_cmd, w_cmd=w_cmd)

        # Render
        fpv_rgb = fpv[:, :, ::-1]
        fpv_surface = pygame.image.frombuffer(fpv_rgb.tobytes(), (W, H), "RGB")

        map_rgb = self.slam.render_topdown_map(map_w, map_h) if self.slam is not None else np.zeros((map_h, map_w, 3), np.uint8)
        map_surface = pygame.image.frombuffer(map_rgb.tobytes(), (map_w, map_h), "RGB")

        self.screen.blit(fpv_surface, (0, 0))
        self.screen.blit(map_surface, (W, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardSLAMPlayer())