package com.example.myapplication.model;

import android.graphics.Rect;

public class BoundingBox {

    public float x1;
    public float y1;
    public float x2;
    public float y2;

    public BoundingBox(float x1, float y1, float x2, float y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    public static BoundingBox fromCenterWidthHeight(float centerX, float centerY, float width, float height) {
        float halfW = width / 2f;
        float halfH = height / 2f;
        return new BoundingBox(centerX - halfW, centerY - halfH, centerX + halfW, centerY + halfH);
    }

    public float width() {
        return x2 - x1;
    }

    public float height() {
        return y2 - y1;
    }

    public float area() {
        return Math.max(0f, width()) * Math.max(0f, height());
    }

    public float centerX() {
        return (x1 + x2) / 2f;
    }

    public float centerY() {
        return (y1 + y2) / 2f;
    }

    public float iou(BoundingBox other) {
        float interX1 = Math.max(x1, other.x1);
        float interY1 = Math.max(y1, other.y1);
        float interX2 = Math.min(x2, other.x2);
        float interY2 = Math.min(y2, other.y2);

        float interW = Math.max(0f, interX2 - interX1);
        float interH = Math.max(0f, interY2 - interY1);
        float interArea = interW * interH;

        float unionArea = area() + other.area() - interArea;
        if (unionArea <= 0f) {
            return 0f;
        }
        return interArea / unionArea;
    }

    public BoundingBox clip(float maxX, float maxY) {
        float cx1 = Math.max(0f, Math.min(x1, maxX));
        float cy1 = Math.max(0f, Math.min(y1, maxY));
        float cx2 = Math.max(0f, Math.min(x2, maxX));
        float cy2 = Math.max(0f, Math.min(y2, maxY));
        return new BoundingBox(cx1, cy1, cx2, cy2);
    }

    public Rect toRect() {
        return new Rect(Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2));
    }

    @Override
    public String toString() {
        return "BoundingBox{x1=" + x1 + ", y1=" + y1 + ", x2=" + x2 + ", y2=" + y2 + "}";
    }
}
