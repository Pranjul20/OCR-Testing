package com.example.myapplication.utils;

import com.example.myapplication.model.Detection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class NMS {

    private NMS() {
    }

    /**
     * Greedy, per-class non-maximum suppression, matching Ultralytics' default NMS behavior.
     */
    public static List<Detection> apply(List<Detection> detections, float iouThreshold) {
        return apply(detections, iouThreshold, false);
    }

    /**
     * Greedy non-maximum suppression. When {@code classAgnostic} is true, overlapping boxes
     * are compared regardless of predicted class — needed for single-digit slots, where the
     * model sometimes emits two different digit classes (e.g. "2" and "5") both overlapping
     * the same box; per-class NMS would keep both since it never compares across classes.
     */
    public static List<Detection> apply(List<Detection> detections, float iouThreshold, boolean classAgnostic) {
        List<Detection> sorted = new ArrayList<>(detections);
        Collections.sort(sorted, (a, b) -> Float.compare(b.confidence, a.confidence));

        boolean[] suppressed = new boolean[sorted.size()];
        List<Detection> kept = new ArrayList<>();

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) {
                continue;
            }
            Detection current = sorted.get(i);
            kept.add(current);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }
                Detection other = sorted.get(j);
                if (!classAgnostic && current.classId != other.classId) {
                    continue;
                }
                if (current.box.iou(other.box) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
    }
}
