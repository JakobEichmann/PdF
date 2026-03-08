public class E01_AssignOutsideInterval_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 100; // EXPECT: FAIL (100 not in [1,3])
    }
}
