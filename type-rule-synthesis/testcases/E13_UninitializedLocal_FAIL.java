public class E13_UninitializedLocal_FAIL {
    public void foo(boolean b) {
        int x;
        if (b) {
            x = 1;
        }
        // x may be uninitialized here
        @Interval(min = 1, max = 1)
        int l = x; // EXPECT: FAIL
    }
}
