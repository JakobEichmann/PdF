/*
annotation EqualTo(int other) int
    :<==> "§subject§ == §other§"
    for "true";
*/

public final class EqualityIdent {

    public void foo(int arg) {
        // :: error: assignment.type.incompatible
        @EqualTo(other="arg") int l0 = arg;
    }

    public void bar(@EqualTo(other="arg1") int arg0, int arg1) {
        // :: error: assignment.type.incompatible
        @EqualTo(other="arg0") int l0 = arg1;
    }
}

/*
Expected result:

---
v : @EqualTo(other="v")

v1 : @EqualTo(other="v2")
---
v2 : @EqualTo(other="v1")
*/
