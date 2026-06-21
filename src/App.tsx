import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import ProjectPage from "./pages/ProjectPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/projets/:slug" element={<ProjectPage />} />
      <Route path="*" element={<Home />} />
    </Routes>
  );
}

export default App;
