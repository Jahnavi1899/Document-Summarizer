import React, {useContext} from "react";
import { SharedContext } from "../SharedContext";

export default function Summary(){
    const { summary } = useContext(SharedContext);
    return (
        <>
            <div className="top-row">
            <h3 className="text-center">Summary</h3>
                <div className="col-md-12" style={{ overflowY: "scroll" }}>
                    <div id="text-summary" style={{ margin: 20 }}>
                        {summary}
                    </div>
                </div>
            </div>
        </>
    )
}