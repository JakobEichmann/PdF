public class E07_AddTwoIntervals_AnnotationMismatch_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    @Interval(min = 10, max = 20)
    int g;

    public void foo() {
        f = 3;
        g = 10;
        // f + g in [11,23], but annotation too narrow
        @Interval(min = 12, max = 22)
        int l = f + g; // EXPECT: FAIL
    }
}
