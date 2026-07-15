package com.example.myapplication.utils;

import com.example.myapplication.model.Detection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class DigitSorter {

    private DigitSorter() {
    }

    public static List<Detection> sortLeftToRight(List<Detection> digits) {
        List<Detection> sorted = new ArrayList<>(digits);
        Collections.sort(sorted, (a, b) -> Float.compare(a.box.x1, b.box.x1));
        return sorted;
    }

    public static String buildTemperatureString(List<Detection> digits) {
        List<Detection> sorted = sortLeftToRight(digits);
        StringBuilder builder = new StringBuilder();
        for (Detection detection : sorted) {
            builder.append(detection.label);
        }
        String raw = builder.toString();

        // 2 digits: whole number, no decimal point (e.g. "25").
        // 3 or 4 digits: last digit is the decimal place, equivalent to inserting a point
        // before (value % 10) (e.g. "258" -> "25.8", "1023" -> "102.3").
        int digitCount = sorted.size();
        if (digitCount == 3 || digitCount == 4) {
            return raw.substring(0, digitCount - 1) + "." + raw.substring(digitCount - 1);
        }
        return raw;
    }
}
