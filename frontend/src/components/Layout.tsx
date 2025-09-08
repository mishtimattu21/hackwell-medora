import { ReactNode } from "react";
import Header from "./Header";
import Footer from "./Footer";

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen flex flex-col page-bg relative overflow-hidden">
      {/* Decorative spotlights */}
      <div className="spotlight spotlight-lg -top-24 -left-24" />
      <div className="spotlight spotlight-md top-1/3 -right-32" />
      <div className="spotlight spotlight-sm bottom-10 left-1/2 -translate-x-1/2" />
      <Header />
      <main className="flex-grow">
        {children}
      </main>
      <Footer />
    </div>
  );
};

export default Layout;