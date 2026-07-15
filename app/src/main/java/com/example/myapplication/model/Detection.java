package com.example.myapplication.model;

public class Detection {

    public final int classId;
    public final String label;
    public final float confidence;
    public final BoundingBox box;

    public Detection(int classId, String label, float confidence, BoundingBox box) {
        this.classId = classId;
        this.label = label;
        this.confidence = confidence;
        this.box = box;
    }

    @Override
    public String toString() {
        return "Detection{label='" + label + "', confidence=" + confidence + ", box=" + box + "}";
    }
}
