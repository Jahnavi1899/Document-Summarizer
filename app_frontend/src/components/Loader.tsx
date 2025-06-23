import React, { useContext } from "react";
import { SharedContext } from "../SharedContext";
import { RotatingLines } from "react-loader-spinner";

const Loader: React.FC = () => {
  const { loader } = useContext(SharedContext)!;
  
  if (!loader) {
    return null;
  }
  
  return (
    <div className="loader-overlay">
      <RotatingLines
        strokeColor="grey"
        strokeWidth="5"
        animationDuration="0.75"
        width="96"
        visible={true}
      />
    </div>
  );
};

export default Loader; 