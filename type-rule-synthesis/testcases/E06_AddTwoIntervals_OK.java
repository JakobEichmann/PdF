public class E06_AddTwoIntervals_OK {
    @Interval(min = 1, max = 3)
    int f;

    @Interval(min = 10, max = 20)
    int g;

    public void foo() {
        f = 3;
        g = 10;
        // f + g in [11,23]
        @Interval(min = 11, max = 23)
        int l = f + g; // EXPECT: OK (needs interval add rule)
    }
}
