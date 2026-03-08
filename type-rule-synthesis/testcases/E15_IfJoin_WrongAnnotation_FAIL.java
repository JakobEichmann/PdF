public class E15_IfJoin_WrongAnnotation_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    public void foo(boolean b) {
        if (b) {
            f = 1;
        } else {
            f = 3;
        }
        // f + 1 in [2,4], annotation too narrow
        @Interval(min = 3, max = 3)
        int l = f + 1; // EXPECT: FAIL
    }
}
