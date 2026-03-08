/*
annotation Interval(int min, int max) int
    :<==> "§subject§ >= §min§ && §subject§ <= §max§"
    for "§min§ >= 0 && §min§ <= §max§";
*/

public final class IntervalIdent {

    public void foo(int arg) {
        // :: error: assignment.type.incompatible
        @Interval(min="arg",max="0x7fffffff") int l0 = arg;
        // :: error: assignment.type.incompatible
        @Interval(min="0xffffffff",max="arg") int l1 = arg;
    }
}

/*
Expected result:

---
v : @Interval(min="v", max="v")

v : @Interval(min="a", max="b")
c <= a
b <= d
---
v : @Interval(min="c", max="d")
*/
