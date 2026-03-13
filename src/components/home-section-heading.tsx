import type { ReactNode } from "react";

interface HomeSectionHeadingProps {
  kicker: string;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function HomeSectionHeading({ kicker, title, description, action }: HomeSectionHeadingProps) {
  return (
    <div className="section-header-row">
      <div className="section-header">
        <span className="section-kicker">{kicker}</span>
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
      {action ? <div className="section-header-action">{action}</div> : null}
    </div>
  );
}
