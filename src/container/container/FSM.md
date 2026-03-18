## ILGP_State.WAIT_TRIGGER:

这个模式下挂起就行，暂时不进行任何操作。

## ILGP_State.INFER_VLM:

这个模式让VLM进行waypiont输出逻辑，将结果可视化在 `/bridge/nav2_goal` ，在RVIZ2里可视化，但是暂时不进行导航。在 `_exec_ilgp_process` 中判断返回的sta，是finshi的话就跳到ILGP_State.WAIT_TRIGGER挂起，是move的话就跳到 `ILGP_State.Moving`

## ILGP_State.Moving:

这个模式等待 `self.ensure_nav2 = True`，然后才进行实际导航。根据导航的结果，成功后进行下一次inferlm
