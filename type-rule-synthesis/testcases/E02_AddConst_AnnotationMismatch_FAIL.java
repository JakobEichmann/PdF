public class E02_AddConst_AnnotationMismatch_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 3;
        // f + 2 in [3,5], but annotation says [2,4]
        @Interval(min = 2, max = 4)
        int l = f + 2; // EXPECT: FAIL
    }
}